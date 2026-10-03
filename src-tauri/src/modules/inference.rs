use crate::modules::models::BBox;
use crate::modules::{
    constants::{COCO_CLASSES, LABEL_PALETTE, MOTION_CLASSES, STATIC_CLASSES},
    drawing::{
        draw_detection, draw_detection_with_mask, draw_motion_overlays, include_font, LabelSet,
    },
    image_utils::load_and_preprocess_image,
    models::{Detection, PreprocessedImage, SegInferenceResult},
    video_utils::{find_motion_pixel, is_video, preprocess_frame},
};

use image::DynamicImage;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter};
use tract_onnx::prelude::tract_ndarray::{Axis, ArrayView2, Ix2, Ix3};
use tract_onnx::prelude::*;
use uuid::Uuid;
use walkdir::WalkDir;

/// Maximum number of detections kept per model run (after NMS for raw layouts).
const MAX_DETECTIONS: usize = 300;

/// Fixed model input size: this app always feeds 640x640 letterboxed images
/// (see `load_and_preprocess_image` and `preprocess_frame`), and all model
/// families in use — stock Ultralytics and the fine-tuned instar models,
/// pixel and normalized e2e variants alike — define their output
/// coordinates relative to that 640x640 letterbox input (see
/// `parse_od_output`).
const INPUT_SIZE: f32 = 640.0;

/// IoU threshold used when NMS is applied by this app (raw, pre-NMS model outputs).
const NMS_IOU_THRESHOLD: f32 = 0.45;

type TractModel = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Load a single ONNX model from the user-selected path and make it runnable.
pub fn load_model(path: &str) -> TractResult<TractModel> {
    tract_onnx::onnx()
        .model_for_path(Path::new(path))?
        .into_optimized()?
        .into_runnable()
}

/// Rescale a box from the 640x640 letterboxed input space back to the
/// original image space, clamping to the image boundaries.
#[inline]
fn to_image_space(
    v: f32,
    pad: u32,
    scale: f32,
    bound: u32,
) -> f32 {
    ((v - pad as f32) / scale).clamp(0.0, bound as f32)
}

fn box_iou(a: &Detection, b: &Detection) -> f32 {
    let ax1 = a.x1.min(a.x2).max(0.0);
    let ay1 = a.y1.min(a.y2).max(0.0);
    let ax2 = a.x1.max(a.x2).max(0.0);
    let ay2 = a.y1.max(a.y2).max(0.0);
    let bx1 = b.x1.min(b.x2).max(0.0);
    let by1 = b.y1.min(b.y2).max(0.0);
    let bx2 = b.x1.max(b.x2).max(0.0);
    let by2 = b.y1.max(b.y2).max(0.0);

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    let ua = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0)
        + (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0)
        - inter;

    if ua <= 0.0 {
        0.0
    } else {
        inter / ua
    }
}

/// Greedy NMS: keep the highest-scoring box, drop all overlapping boxes
/// below `NMS_IOU_THRESHOLD`, repeat. Mutates `detections` in place.
fn nms_inplace(detections: &mut Vec<Detection>) {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let n = detections.len();
    let mut kept = vec![true; n];

    for i in 0..n {
        if !kept[i] {
            continue;
        }
        for j in (i + 1)..n {
            if kept[j] && box_iou(&detections[i], &detections[j]) > NMS_IOU_THRESHOLD {
                kept[j] = false;
            }
        }
    }

    *detections = detections
        .iter()
        .zip(kept.iter())
        .filter_map(|(d, k)| k.then(|| d.clone()))
        .take(MAX_DETECTIONS)
        .collect();
}

/// Parse an object-detection output tensor (batch dim already removed) into
/// detections in image space.
///
/// Two tensor conventions exist for YOLO-style ONNX exports:
///
///  1. "Pre-NMS / raw" : shape (4 + nc features, A anchors), e.g. [84, 8400]
///     for COCO. Feature-major: rows 0..4 are **center-x, center-y,
///     width, height** (NOT x1,y1,x2,y2 — that is the classic Ultralytics
///     raw export format), class confidences in rows 4..(4+nc). Needs NMS.
///
///  2. "End-to-end (NMS baked in)" : shape (N, 6) with N detections (e.g.
///     300). Row-major: [x1, y1, x2, y2, score, class_id] per row.
///     xyxy is ALWAYS relative to the 640x640 letterbox input, emitted two
///     ways (discriminated by magnitude, see the e2e branch below):
///       - pixel space: xyxy in 640-letterbox pixels (fine-tuned YOLO26
///         e2e exports);
///       - normalized [0..1]: the SAME letterbox xyxy pre-divided by 640
///         (yolov10s / static e2e exports). A model that only receives the
///         640 square cannot know the original image size, so the
///         normalized form is relative to the letterbox, never the
///         original image.
///     Both are decoded with the recorded (scale, pad) triple, exactly like
///     the raw branch.
///
/// Heuristic distinction: raw outputs have far more anchors than features,
/// so rows < cols for raw and rows > cols for end-to-end outputs.
fn parse_od_output(
    det: ArrayView2<f32>,
    conf_threshold: f32,
    orig_w: u32,
    orig_h: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
) -> Vec<Detection> {
    let (rows, cols) = (det.shape()[0], det.shape()[1]);

    if rows < cols {
        let nc = (rows - 4).max(1);
        let mut final_detections = Vec::new();

        for a in 0..cols {
            let mut best_class = 0usize;
            let mut best_score = -1.0f32;

            for c in 0..nc {
                let s = det[[4 + c, a]];
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }

            if best_score < conf_threshold {
                continue;
            }

            final_detections.push(Detection {
                // Raw export is center-x, center-y, width, height.
                x1: to_image_space(det[[0, a]] - det[[2, a]] / 2.0, pad_x, scale, orig_w),
                y1: to_image_space(det[[1, a]] - det[[3, a]] / 2.0, pad_y, scale, orig_h),
                x2: to_image_space(det[[0, a]] + det[[2, a]] / 2.0, pad_x, scale, orig_w),
                y2: to_image_space(det[[1, a]] + det[[3, a]] / 2.0, pad_y, scale, orig_h),
                score: best_score,
                class_id: best_class,
                mask_coefficients: None,
            });
        }

        // Raw outputs have not gone through NMS, so de-duplicate here.
        nms_inplace(&mut final_detections);
        final_detections
    } else {
        // All e2e boxes are xyxy relative to the 640x640 LETTERBOXED model
        // input — never to the original image (a model that receives only
        // the 640 square cannot know the original size):
        //   - pixel exports emit letterbox pixels directly;
        //   - normalized exports pre-divide the SAME letterbox coords by 640.
        // Measured on screenshot.png (3840x2182) against the letterbox-space
        // COCO reference: the same person decodes to [313.9,230.3,399.4,436.7]
        // (fine-tuned pixel e2e) vs [313.7,231,400.5,437.7] (COCO raw), and
        // the same car to [0.379,0.326,0.877,0.611]*640 (fine-tuned
        // normalized e2e) vs [240,208,560,394] (COCO raw).
        //
        // Normalized coords are ≤ 1.0 by definition, pixel coords are tens
        // of px at least — the max magnitude discriminates the two.
        let mut max_coord = 0.0f32;
        for i in 0..rows {
            for c in 0..4 {
                max_coord = max_coord.max(det[[i, c]].abs());
            }
        }
        let normalized = max_coord <= 1.5;

        // Map into 640-letterbox space, then un-letterbox with the recorded
        // (scale, pad) triple — identical to the raw branch.
        let to_img = |v: f32, horiz: bool| -> f32 {
            let lb = if normalized { v.clamp(0.0, 1.0) * INPUT_SIZE } else { v };
            let pad = if horiz { pad_x } else { pad_y };
            let bound = if horiz { orig_w } else { orig_h };
            to_image_space(lb, pad, scale, bound)
        };

        (0..rows)
            .filter(|&i| det[[i, 4]] >= conf_threshold)
            .map(|i| Detection {
                x1: to_img(det[[i, 0]], true),
                y1: to_img(det[[i, 1]], false),
                x2: to_img(det[[i, 2]], true),
                y2: to_img(det[[i, 3]], false),
                score: det[[i, 4]],
                class_id: det[[i, 5]] as usize,
                mask_coefficients: None,
            })
            .collect()
    }
}

fn run_od_inference(
    model: &TractModel,
    input_tensor: Tensor,
    conf_threshold: f32,
    orig_w: u32,
    orig_h: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
) -> TractResult<Vec<Detection>> {
    let result = model.run(tvec!(input_tensor.into()))?;

    let output = result[0].to_array_view::<f32>()?;
    let det =
        output.index_axis(Axis(0), 0).into_dimensionality::<Ix2>()?; // 2D: [rows, cols]

    Ok(parse_od_output(det, conf_threshold, orig_w, orig_h, scale, pad_x, pad_y))
}

fn run_seg_inference(
    model: &TractModel,
    input_tensor: Tensor,
    conf_threshold: f32,
    orig_w: u32,
    orig_h: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
) -> TractResult<SegInferenceResult> {
    let result = model.run(tvec!(input_tensor.into()))?;

    let boxes_raw = result[0].to_array_view::<f32>()?;
    let masks_raw = result[1].to_array_view::<f32>()?;

    let boxes = boxes_raw
        .index_axis(Axis(0), 0)
        .into_dimensionality::<Ix2>()?; // 2D: [rows, cols]
    let proto_masks = masks_raw
        .index_axis(Axis(0), 0)
        .into_dimensionality::<Ix3>()?
        .to_owned();

    let detections = parse_seg_output(boxes, conf_threshold, orig_w, orig_h, scale, pad_x, pad_y)?;

    Ok(SegInferenceResult {
        detections,
        proto_masks,
    })
}

/// Parse a segmentation-boxes tensor (batch dim already removed) into
/// detections with 32 mask coefficients each, in image space.
///
/// Both tensor conventions from `parse_od_output` exist here:
///
///  - Raw / pre-NMS: (4 + nc + 32 features, A anchors), e.g. [116, 8400] for
///    COCO: rows 0..4 are center-x, center-y, width, height, rows 4..(4+nc)
///    class confidences, then 32 mask coefficients. Needs NMS.
///  - End-to-end (N, 6 + 32), e.g. [300, 38]:
///    [x1, y1, x2, y2, score, class_id, coeff_0 .. coeff_31] per row.
fn parse_seg_output(
    boxes: ArrayView2<f32>,
    conf_threshold: f32,
    orig_w: u32,
    orig_h: u32,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
) -> TractResult<Vec<Detection>> {
    let (rows, cols) = (boxes.shape()[0], boxes.shape()[1]);
    let mut detections = Vec::new();

    if rows < cols {
        // Raw / pre-NMS layout: (4 + nc + 32, A)
        //   rows 0..4        : box (center-x, center-y, width, height)
        //   rows 4..(4+nc)   : class confidences (one row per class)
        //   rows (4+nc)..end : 32 mask coefficients
        let n_coeff = rows - 4;
        if n_coeff < 33 {
            return Err(anyhow::anyhow!(
                "Unsupported raw segmentation output: expected 4 + nc + 32 features"
            ));
        }

        let nc = n_coeff - 32;
        let coeff_start = 4 + nc;

        for a in 0..cols {
            let mut best_class = 0usize;
            let mut best_score = -1.0f32;

            for c in 0..nc {
                let s = boxes[[4 + c, a]];
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }

            if best_score < conf_threshold {
                continue;
            }

            let mut coeffs = [0.0f32; 32];
            for idx in 0..32 {
                coeffs[idx] = boxes[[coeff_start + idx, a]];
            }

            detections.push(Detection {
                // Raw export is center-x, center-y, width, height.
                x1: to_image_space(boxes[[0, a]] - boxes[[2, a]] / 2.0, pad_x, scale, orig_w),
                y1: to_image_space(boxes[[1, a]] - boxes[[3, a]] / 2.0, pad_y, scale, orig_h),
                x2: to_image_space(boxes[[0, a]] + boxes[[2, a]] / 2.0, pad_x, scale, orig_w),
                y2: to_image_space(boxes[[1, a]] + boxes[[3, a]] / 2.0, pad_y, scale, orig_h),
                score: best_score,
                class_id: best_class,
                mask_coefficients: Some(coeffs),
            });
        }

        nms_inplace(&mut detections);
    } else {
        // End-to-end layout: (N, 6 + 32)
        //   [x1, y1, x2, y2, score, class_id, coeff_0 .. coeff_31]
        //
        // x1..y2 are relative to the 640x640 letterbox input (pixel e2e
        // exports emit letterbox pixels; normalized exports pre-divide the
        // same letterbox coords by 640). Same convention as OD — see the
        // e2e branch of `parse_od_output` for details and measurements.
        // The mask prototype grid is in the same letterbox space (verified:
        // COCO and fine-tuned seg models localize the same person at grid
        // rows ~50-110 for a box at lb-rows 58-109), which is exactly what
        // `draw_detection_with_mask` samples.
        let mut max_coord = 0.0f32;
        for i in 0..rows {
            for c in 0..4 {
                max_coord = max_coord.max(boxes[[i, c]].abs());
            }
        }
        let normalized = max_coord <= 1.5;

        let to_img = |v: f32, horiz: bool| -> f32 {
            let lb = if normalized { v.clamp(0.0, 1.0) * INPUT_SIZE } else { v };
            let pad = if horiz { pad_x } else { pad_y };
            let bound = if horiz { orig_w } else { orig_h };
            to_image_space(lb, pad, scale, bound)
        };

        for i in 0..rows {
            let score = boxes[[i, 4]];

            if score < conf_threshold {
                continue;
            }

            let mut coeffs = [0.0f32; 32];
            for idx in 0..32 {
                coeffs[idx] = boxes[[i, 6 + idx]];
            }

            detections.push(Detection {
                x1: to_img(boxes[[i, 0]], true),
                y1: to_img(boxes[[i, 1]], false),
                x2: to_img(boxes[[i, 2]], true),
                y2: to_img(boxes[[i, 3]], false),
                score,
                class_id: boxes[[i, 5]] as usize,
                mask_coefficients: Some(coeffs),
            });
        }
    }

    Ok(detections)
}

/// Build the ordered (name, color) label table from a built-in class list.
fn const_labels<const N: usize>(classes: &[( &str, [u8; 3]); N]) -> LabelSet {
    classes.iter().map(|(name, color)| (name.to_string(), *color)).collect()
}

/// Parse class names out of a user-supplied label file.
///
/// Accepts the common shapes:
///  - one name per line (`person`, `car`, ...)
///  - YAML flow list: `names: [ 'person', 'car', ... ]`
///  - YAML map: `names:` followed by `0: person`, `1: car`, ...
/// Whitespace is trimmed and quotes are stripped. Blank lines and `#` comments
/// are ignored. Returns an empty Vec if no names are found.
fn parse_label_names(content: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();

    for raw in content.lines() {
        let line = raw.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Flow list: `names: [ 'a', 'b' ]` or `['a', 'b']`.
        if let Some(open) = line.find('[') {
            if let Some(close) = line[open..].find(']') {
                // Content is strictly between the brackets: `[` is at `open`,
                // `]` at `open + close` (close is the offset within `line[open..]`).
                let inner = &line[open + 1..open + close];

                for part in inner.split(',') {
                    let name = part.trim().trim_matches(|c| c == '\'' || c == '"').trim();
                    if !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
                continue;
            }
        }

        // Any `key: value` line is YAML. A numeric key (`0: person`) is a
        // class name; a bare `names:` header or metadata (`nc: 19`) is skipped.
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            if key != "names" && key.parse::<usize>().is_ok() {
                let name = value.trim().trim_matches(|c| c == '\'' || c == '"').trim();
                if !name.is_empty() {
                    names.push(name.to_string());
                }
            }
            continue;
        }

        // One name per line (no colon, no brackets).
        names.push(line.to_string());
    }

    names
}

/// Resolve the selected label-set descriptor into a concrete ordered label
/// table for **one specific model**. `label_set` is one of: "coco", "motion",
/// "static", or a file path to a custom label file. An empty/None/"auto"
/// descriptor is resolved per model: the file name is inspected and a
/// built-in table is suggested (a name containing "static" → Static,
/// "motion" → Motion, otherwise COCO — the safe default for stock
/// Ultralytics exports).
fn resolve_labels(label_set: Option<&str>, model_path: &str) -> Result<(LabelSet, String), String> {
    let descriptor = label_set.unwrap_or("").trim().to_lowercase();

    // Auto: let the model name drive the suggestion.
    if descriptor.is_empty() || descriptor == "auto" {
        let name = model_path.to_lowercase();
        let model_name = model_path.rsplit('/').next().unwrap_or(model_path);
        if name.contains("static") {
            let desc = format!("Static (26 classes) — suggested from \"{model_name}\"");
            return Ok((const_labels(&STATIC_CLASSES), desc));
        }
        if name.contains("motion") {
            let desc = format!("Motion (19 classes) — suggested from \"{model_name}\"");
            return Ok((const_labels(&MOTION_CLASSES), desc));
        }
        let desc = format!("COCO (80 classes) — suggested from \"{model_name}\"");
        return Ok((const_labels(&COCO_CLASSES), desc));
    }

    match descriptor.as_str() {
        "coco" => Ok((const_labels(&COCO_CLASSES), "COCO (80 classes)".to_string())),
        "motion" => Ok((const_labels(&MOTION_CLASSES), "Motion (19 classes)".to_string())),
        "static" => Ok((const_labels(&STATIC_CLASSES), "Static (26 classes)".to_string())),

        // Everything else is treated as a path to a custom label file.
        path => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read label file {path}: {e}"))?;

            let names = parse_label_names(&content);
            if names.is_empty() {
                return Err(format!(
                    "No class names parsed from label file {path}. Expected one name per line, \
                     a `names: ['a', 'b']` list, or `0: name` YAML entries."
                ));
            }

            // Custom files carry no colors, so cycle the palette.
            let count = names.len();
            let labels = names
                .into_iter()
                .enumerate()
                .map(|(i, name)| (name, LABEL_PALETTE[i % LABEL_PALETTE.len()]))
                .collect::<LabelSet>();

            Ok((
                labels,
                format!("Custom file ({} classes): {}", count, path),
            ))
        }
    }
}

/// Build a single log line with the top-5 detections (by score) for a model,
/// resolved against `labels` so the UI log shows real class names.
fn top5_log_line(model_tag: &str, detections: &[Detection], labels: &LabelSet) -> String {
    if detections.is_empty() {
        return format!("{model_tag}: 0 detections");
    }

    let mut sorted: Vec<&Detection> = detections.iter().collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top = sorted
        .iter()
        .take(5)
        .map(|d| {
            let name = labels
                .get(d.class_id)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| format!("class {}", d.class_id));
            format!("{name} {:.2}", d.score)
        })
        .collect::<Vec<_>>()
        .join(", ");

    format!("{model_tag}: {} detection(s); top-5: {top}", detections.len())
}

fn process_frame(
    prep: &PreprocessedImage,
    od_model: Arc<TractModel>,
    seg_model: Arc<TractModel>,
    conf_threshold: f32,
    motion_boxes: Option<&[BBox]>,
    output_dir: &Path,
    app: &AppHandle,
    video_prep_ms: Option<u128>,
    od_labels: &LabelSet,
    seg_labels: &LabelSet,
) {
    let font = include_font();

    let mut od_canvas = prep.original_img.to_rgb8();
    let mut seg_canvas = prep.original_img.to_rgb8();

    let ((od_result, od_time_ms), (seg_result, seg_time_ms)) = rayon::join(
        || {
            let start = Instant::now();

            let result = run_od_inference(
                &od_model,
                prep.input_tensor.clone(),
                conf_threshold,
                prep.orig_w,
                prep.orig_h,
                prep.scale,
                prep.pad_x,
                prep.pad_y,
            );

            (result, start.elapsed().as_millis())
        },
        || {
            let start = Instant::now();
            let result = run_seg_inference(
                &seg_model,
                prep.input_tensor.clone(),
                conf_threshold,
                prep.orig_w,
                prep.orig_h,
                prep.scale,
                prep.pad_x,
                prep.pad_y,
            );

            (result, start.elapsed().as_millis())
        },
    );

    let od_detections = od_result.unwrap_or_default();

    let _ = app.emit(
        "log_event",
        top5_log_line("OD", &od_detections, od_labels),
    );

    for det in &od_detections {
        draw_detection(&mut od_canvas, det, font, od_labels);
    }

    if let Ok(seg_result) = seg_result {
        let _ = app.emit(
            "log_event",
            top5_log_line("SEG", &seg_result.detections, seg_labels),
        );

        for det in &seg_result.detections {
            draw_detection_with_mask(
                &mut seg_canvas,
                det,
                font,
                &seg_result.proto_masks,
                prep.scale,
                prep.pad_x,
                prep.pad_y,
                seg_labels,
            );
        }
    }

    if let Some(boxes) = motion_boxes {
        let mut od_rgba = DynamicImage::ImageRgb8(od_canvas.clone()).to_rgba8();

        let mut seg_rgba = DynamicImage::ImageRgb8(seg_canvas.clone()).to_rgba8();

        draw_motion_overlays(&mut od_rgba, boxes);
        draw_motion_overlays(&mut seg_rgba, boxes);

        od_canvas = DynamicImage::ImageRgba8(od_rgba).to_rgb8();
        seg_canvas = DynamicImage::ImageRgba8(seg_rgba).to_rgb8();
    }

    let uuid = Uuid::new_v4().to_string();

    let od_path = output_dir.join(format!("od_{}.jpg", uuid));
    let seg_path = output_dir.join(format!("seg_{}.jpg", uuid));

    let _ = od_canvas.save(&od_path);
    let _ = seg_canvas.save(&seg_path);

    // Get the absolute path to ensure the asset protocol can resolve it
    let od_abs = std::fs::canonicalize(&od_path).unwrap_or(od_path.clone());
    let seg_abs = std::fs::canonicalize(&seg_path).unwrap_or(seg_path.clone());

    let _ = app.emit(
        "image_event",
        serde_json::json!({
            "od_path": od_abs.to_string_lossy(),
            "seg_path": seg_abs.to_string_lossy(),
            "od_time_ms": od_time_ms,
            "seg_time_ms": seg_time_ms,
            "video_prep_ms": video_prep_ms.unwrap_or(0),
        }),
    );
}

#[tauri::command]
pub async fn run_inference(
    input_dir: String,
    od_model_path: String,
    seg_model_path: String,
    conf_threshold: f32,
    // One label descriptor per model: "coco" | "motion" | "static" | "auto"
    // | "" (per-model name-based suggestion), or a path to a custom label
    // file. Each model resolves independently, so mixed datasets (e.g. a
    // COCO OD model compared against a motion SEG model) work.
    od_labels: Option<String>,
    seg_labels: Option<String>,
    app: AppHandle,
) {
    // Model loading + inference are CPU heavy, so run them off the main thread.
    let app_bg = app.clone();
    tokio::task::spawn_blocking(move || {
        let log = |app: &AppHandle, message: String| {
            let _ = app.emit("log_event", message);
        };

        // Resolve each model's label table up front so a bad descriptor
        // aborts before any heavy model-loading or per-file work.
        let od_labels = match resolve_labels(od_labels.as_deref(), &od_model_path) {
            Ok((labels, description)) => {
                log(&app_bg, format!("OD label set: {description}"));
                labels
            }
            Err(e) => {
                log(&app_bg, e);
                return;
            }
        };

        let seg_labels = match resolve_labels(seg_labels.as_deref(), &seg_model_path) {
            Ok((labels, description)) => {
                log(&app_bg, format!("SEG label set: {description}"));
                labels
            }
            Err(e) => {
                log(&app_bg, e);
                return;
            }
        };

        // Lazily load the models selected in the UI on every run.
        let od_model = match load_model(&od_model_path) {
            Ok(model) => {
                log(&app_bg, format!("Loaded OD model: {}", od_model_path));
                Arc::new(model)
            }
            Err(e) => {
                log(
                    &app_bg,
                    format!("Failed to load OD model {od_model_path}: {e}"),
                );
                return;
            }
        };

        let seg_model = match load_model(&seg_model_path) {
            Ok(model) => {
                log(&app_bg, format!("Loaded SEG model: {}", seg_model_path));
                Arc::new(model)
            }
            Err(e) => {
                log(
                    &app_bg,
                    format!("Failed to load SEG model {seg_model_path}: {e}"),
                );
                return;
            }
        };

        let input_dir = PathBuf::from(&input_dir);
        let output_dir = PathBuf::from("./predictions");
        if !output_dir.exists() {
            let _ = std::fs::create_dir_all(&output_dir);
        }

        let entries: Vec<PathBuf> = WalkDir::new(&input_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_path_buf())
            .collect();

        log(&app_bg, format!("Found {} input file(s) in {}", entries.len(), input_dir.display()));

        entries.par_iter().for_each(|path| {
            if is_video(path) {
                let video_processing_start = Instant::now();
                log(&app_bg, format!("Starting video analysis: {}", path.display()));
                if let Some((motion_boxes, frame)) = find_motion_pixel(path, 3, -1, 5, 10) {
                    log(
                        &app_bg,
                        format!(
                            "Motion found in video! Processing frame with {} motion region(s).",
                            motion_boxes.len()
                        ),
                    );
                    let prep = preprocess_frame(&frame, 640);
                    let video_prep_ms = video_processing_start.elapsed().as_millis();

                    process_frame(
                        &prep,
                        od_model.clone(),
                        seg_model.clone(),
                        conf_threshold,
                        Some(&motion_boxes),
                        &output_dir,
                        &app_bg,
                        Some(video_prep_ms),
                        &od_labels,
                        &seg_labels,
                    );
                } else {
                    log(
                        &app_bg,
                        format!("Finished scanning video {}, but no motion was detected.", path.display()),
                    );
                }
            } else {
                if let Ok(prep) = load_and_preprocess_image(path, 640) {
                    log(&app_bg, format!("Starting image analysis: {}", path.display()));
                    process_frame(
                        &prep,
                        od_model.clone(),
                        seg_model.clone(),
                        conf_threshold,
                        None,
                        &output_dir,
                        &app_bg,
                        None, // No video prep time
                        &od_labels,
                        &seg_labels,
                    );
                } else {
                    log(
                        &app_bg,
                        format!("Skipped {} (not a decodable image).", path.display()),
                    );
                }
            }
        });

        log(&app_bg, "Inference complete.".to_string());
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_onnx::prelude::tract_ndarray::Array2;

    const CONF: f32 = 0.45;
    const IMG: u32 = 640;

    /// scale = 1.0, no padding: 640-space coordinates map to themselves.
    fn identity_geom() -> (u32, u32, f32, u32, u32) {
        (IMG, IMG, 1.0, 0, 0)
    }

    fn make_det(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        score: f32,
        class_id: usize,
    ) -> Detection {
        Detection {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
            mask_coefficients: None,
        }
    }

    #[test]
    fn parse_motion_style_yaml_flow_list() {
        let content = "\
# number of classes
nc: 19
# Classes
names: [ 'person', 'car', 'motor', 'train', 'truck', 'dog', 'cat', 'van', 'bird', 'deer', 'bus', 'plate', 'face', 'bike', 'plane', 'boat', 'helicopter', 'horse', 'cow' ]
";
        let labels = parse_label_names(content);
        assert_eq!(labels.len(), 19);
        assert_eq!(labels[0], "person");
        assert_eq!(labels[18], "cow");
    }

    #[test]
    fn parse_static_style_yaml_block_map() {
        let content = "\
# Classes
names:
  0: doors
  1: windows
  25: fire
";
        let labels = parse_label_names(content);
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], "doors");
        assert_eq!(labels[2], "fire");
    }

    #[test]
    fn parse_plain_one_per_line_and_unquoted_flow() {
        assert_eq!(
            parse_label_names("alpha\nbeta\n# comment\n\ngamma\n").len(),
            3
        );
        assert_eq!(
            parse_label_names("names: [ alpha, beta ]\nnc: 2\n").len(),
            2
        );
    }

    #[test]
    fn resolve_labels_builtins_and_auto_suggestion() {
        assert_eq!(resolve_labels(Some("coco"), "yolo26n.onnx").unwrap().0.len(), 80);
        assert_eq!(
            resolve_labels(Some("motion"), "m.onnx").unwrap().0.len(),
            19
        );
        assert_eq!(
            resolve_labels(Some("static"), "s.onnx").unwrap().0.len(),
            26
        );
        assert_eq!(
            resolve_labels(Some("motion"), "m.onnx").unwrap().0[3].0,
            "train"
        );
        assert_eq!(
            resolve_labels(Some("static"), "s.onnx").unwrap().0[25].0,
            "fire"
        );

        // Per-model auto suggestion (None / "" / "auto" all behave the same).
        for descriptor in [None, Some(""), Some("auto")] {
            assert_eq!(
                resolve_labels(descriptor, "models/YOLO26n-seg-motion-600e.onnx")
                    .unwrap()
                    .0
                    .len(),
                19
            );
            assert_eq!(
                resolve_labels(descriptor, "models/YOLO26s-static-600e.onnx")
                    .unwrap()
                    .0
                    .len(),
                26
            );
            // A stock COCO model falls back to the 80-class default.
            assert_eq!(
                resolve_labels(descriptor, "models/yolo26n.onnx").unwrap().0.len(),
                80
            );
        }
    }

    #[test]
    fn top5_log_line_sorts_and_names_classes() {
        let labels = vec![
            ("alpha".to_string(), [1, 2, 3]),
            ("beta".to_string(), [4, 5, 6]),
        ];
        let dets = vec![
            make_det(0.0, 0.0, 1.0, 1.0, 0.5, 1),
            make_det(0.0, 0.0, 1.0, 1.0, 0.9, 0),
        ];
        let line = top5_log_line("OD", &dets, &labels);
        assert!(line.starts_with("OD: 2 detection(s); top-5: alpha 0.90,"), "{line}");
        assert!(line.contains("beta 0.50"));
        assert_eq!(top5_log_line("OD", &[], &labels), "OD: 0 detections");
    }

    #[test]
    fn to_image_space_applies_letterbox_unscale_and_clamps() {
        // 640-space coord 110, pad 10, scale 0.5 -> original-space 200.
        assert_eq!(to_image_space(110.0, 10, 0.5, 1280), 200.0);
        // Coordinates inside the letterbox pad are clamped to 0.
        assert_eq!(to_image_space(5.0, 10, 0.5, 1280), 0.0);
        // Coordinates beyond the image are clamped to the bound.
        assert_eq!(to_image_space(4096.0, 0, 1.0, 640), 640.0);
    }

    #[test]
    fn nms_keeps_best_of_overlapping_boxes_and_distinct_boxes() {
        let mut dets = vec![
            make_det(0.0, 0.0, 100.0, 100.0, 0.7, 0),
            make_det(5.0, 5.0, 105.0, 105.0, 0.92, 1), // overlaps the first
            make_det(200.0, 200.0, 300.0, 300.0, 0.8, 2), // no overlap
        ];

        nms_inplace(&mut dets);

        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].class_id, 1);
        assert!((dets[0].score - 0.92).abs() < 1e-6);
        assert_eq!(dets[1].class_id, 2);
    }

    #[test]
    fn od_raw_layout_thresholds_argmax_and_nms() {
        // (4 boxes + 2 classes) x 12 anchors.
        // cols must be > rows so the layout is recognized as "raw" (real
        // raw exports, e.g. [84, 8400], always have far more anchors).
        let mut det = Array2::<f32>::zeros((6, 12));

        // Anchor 0: class 1 wins argmax with 0.9. raw layout is cx, cy, w, h.
        det[[0, 0]] = 200.0; // center-x
        det[[1, 0]] = 160.0; // center-y
        det[[2, 0]] = 100.0; // width  -> x1 = 150, x2 = 250
        det[[3, 0]] = 80.0; // height -> y1 = 120, y2 = 200
        det[[4, 0]] = 0.1;
        det[[5, 0]] = 0.9;

        // Anchor 1: overlapping box, lower score -> must be NMS'd.
        det[[0, 1]] = 205.0;
        det[[1, 1]] = 165.0;
        det[[2, 1]] = 100.0;
        det[[3, 1]] = 80.0;
        det[[4, 1]] = 0.2;
        det[[5, 1]] = 0.8;

        // Anchor 2: below threshold -> dropped.
        det[[0, 2]] = 10.0;
        det[[1, 2]] = 10.0;
        det[[2, 2]] = 50.0;
        det[[3, 2]] = 50.0;
        det[[4, 2]] = 0.4;
        det[[5, 2]] = 0.0;

        let (orig_w, orig_h, scale, pad_x, pad_y) = identity_geom();
        let dets = parse_od_output(det.view(), CONF, orig_w, orig_h, scale, pad_x, pad_y);

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        assert!((dets[0].score - 0.9).abs() < 1e-6);
        assert!((dets[0].x1 - 150.0).abs() < 1e-6);
        assert!((dets[0].y1 - 120.0).abs() < 1e-6);
        assert!((dets[0].x2 - 250.0).abs() < 1e-6);
        assert!((dets[0].y2 - 200.0).abs() < 1e-6);
        assert!(dets[0].mask_coefficients.is_none());
    }

    #[test]
    fn od_e2e_layout_reads_score_and_class_id() {
        // (9 detections, 6 cols): [x1, y1, x2, y2, score, class_id] per row.
        // rows must be > cols so the layout is recognized as "end-to-end"
        // (real e2e exports are e.g. [300, 6]).
        let mut det = Array2::<f32>::zeros((9, 6));

        det[[0, 0]] = 10.0;
        det[[0, 1]] = 20.0;
        det[[0, 2]] = 100.0;
        det[[0, 3]] = 200.0;
        det[[0, 4]] = 0.8;
        det[[0, 5]] = 3.0;

        det[[1, 0]] = 0.0;
        det[[1, 1]] = 0.0;
        det[[1, 2]] = 1.0;
        det[[1, 3]] = 1.0;
        det[[1, 4]] = 0.1; // below threshold
        det[[1, 5]] = 0.0;

        let (orig_w, orig_h, scale, pad_x, pad_y) = identity_geom();
        let dets = parse_od_output(det.view(), CONF, orig_w, orig_h, scale, pad_x, pad_y);

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 3);
        assert!((dets[0].score - 0.8).abs() < 1e-6);
        assert!(dets[0].mask_coefficients.is_none());
    }

    #[test]
    fn od_e2e_normalized_coords_are_letterbox_over_640() {
        // e2e exports may emit xyxy normalized into [0,1] — but relative to
        // the 640x640 LETTERBOXED input, not the original image: a model
        // receiving only the 640 square cannot know the original size. So
        // decoding is `v * 640 -> (v - pad) / scale`, not `v * orig_dim`.
        // rows must be > cols so this is recognized as "end-to-end"
        // (real e2e exports look like [300, 6]).
        let mut det = Array2::<f32>::zeros((9, 6));

        det[[0, 0]] = 0.2376;
        det[[0, 1]] = 0.2415;
        det[[0, 2]] = 0.8541;
        det[[0, 3]] = 0.5467;
        det[[0, 4]] = 0.97;
        det[[0, 5]] = 1.0;

        det[[1, 4]] = 0.01; // below threshold

        // 1280x720 frame letterboxed to 640x640: scale 0.5, pad_x 0, pad_y 140.
        let dets = parse_od_output(det.view(), CONF, 1280, 720, 0.5, 0, 140);

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        assert!((dets[0].score - 0.97).abs() < 1e-6);
        // letterbox coords: [152.064, 154.56, 546.624, 349.888]
        assert!((dets[0].x1 - 152.064 / 0.5).abs() < 1e-3); // 304.128
        assert!((dets[0].y1 - (154.56 - 140.0) / 0.5).abs() < 1e-3); // 29.12
        assert!((dets[0].x2 - 546.624 / 0.5).abs() < 1e-3); // 1093.248
        assert!((dets[0].y2 - (349.888 - 140.0) / 0.5).abs() < 1e-3); // 419.776
    }

    #[test]
    fn od_e2e_pixel_coords_unletterbox_like_raw_outputs() {
        // Real-world check: YOLO26n-seg-motion (fine-tuned, e2e pixel
        // variant) reports the person in screenshot.png (3840x2182) as
        // [313.9, 230.3, 399.4, 436.7] in letterbox-pixel space — the same
        // object the letterbox-space COCO model reports as
        // [313.7, 231.2, 400.5, 437.7]. Both must decode to nearly the same
        // original-space box, so the e2e box must be un-letterboxed with
        // (v - pad) / scale exactly like the raw branch.
        let mut det = Array2::<f32>::zeros((9, 6));
        det[[0, 0]] = 313.9;
        det[[0, 1]] = 230.3;
        det[[0, 2]] = 399.4;
        det[[0, 3]] = 436.7;
        det[[0, 4]] = 0.97;
        det[[0, 5]] = 0.0;
        det[[1, 4]] = 0.01; // below threshold

        // 3840x2182: scale 1/6, pad_x 0, pad_y 138.
        let dets = parse_od_output(det.view(), CONF, 3840, 2182, 1.0 / 6.0, 0, 138);

        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        // Expected original space: x [1883.4, 2396.4], y [553.8, 1792.2]
        assert!((d.x1 - 1883.4).abs() < 1.0, "x1={}", d.x1);
        assert!((d.y1 - 553.8).abs() < 1.0, "y1={}", d.y1);
        assert!((d.x2 - 2396.4).abs() < 1.0, "x2={}", d.x2);
        assert!((d.y2 - 1792.2).abs() < 1.0, "y2={}", d.y2);
    }

    #[test]
    fn seg_raw_layout_reads_coeffs_from_feature_rows() {
        // (4 boxes + 3 classes + 32 coeffs) x 40 anchors.
        // cols must be > rows so the layout is recognized as "raw" (real raw
        // seg exports are e.g. [116, 8400]).
        let mut boxes = Array2::<f32>::zeros((4 + 3 + 32, 40));

        boxes[[0, 0]] = 10.0;
        boxes[[1, 0]] = 20.0;
        boxes[[2, 0]] = 110.0;
        boxes[[3, 0]] = 220.0;
        boxes[[4, 0]] = 0.05;
        boxes[[5, 0]] = 0.85; // class 1 wins
        boxes[[6, 0]] = 0.30;
        for i in 0..32 {
            boxes[[4 + 3 + i, 0]] = i as f32 * 0.25;
        }

        // Anchor 1: below threshold.
        boxes[[4, 1]] = 0.1;
        boxes[[5, 1]] = 0.0;

        let (orig_w, orig_h, scale, pad_x, pad_y) = identity_geom();
        let dets = parse_seg_output(boxes.view(), CONF, orig_w, orig_h, scale, pad_x, pad_y)
            .expect("raw seg layout should parse");

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 1);
        assert!((dets[0].score - 0.85).abs() < 1e-6);
        let coeffs = dets[0].mask_coefficients.expect("coefficients present");
        for i in 0..32 {
            assert!((coeffs[i] - i as f32 * 0.25).abs() < 1e-6, "coeff {i}");
        }
    }

    #[test]
    fn seg_e2e_layout_reads_coeffs_from_columns() {
        // (9 detections, 6 + 32 cols).
        // rows must be > cols so the layout is recognized as "end-to-end"
        // (real e2e seg exports are e.g. [300, 38], N typically 300).
        let mut boxes = Array2::<f32>::zeros((40, 38));

        boxes[[0, 0]] = 0.0;
        boxes[[0, 1]] = 0.0;
        boxes[[0, 2]] = 100.0;
        boxes[[0, 3]] = 100.0;
        boxes[[0, 4]] = 0.9;
        boxes[[0, 5]] = 2.0;
        for i in 0..32 {
            boxes[[0, 6 + i]] = i as f32;
        }

        boxes[[1, 4]] = 0.05; // below threshold

        let (orig_w, orig_h, scale, pad_x, pad_y) = identity_geom();
        let dets = parse_seg_output(boxes.view(), CONF, orig_w, orig_h, scale, pad_x, pad_y)
            .expect("e2e seg layout should parse");

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 2);
        assert!((dets[0].score - 0.9).abs() < 1e-6);
        let coeffs = dets[0].mask_coefficients.expect("coefficients present");
        for i in 0..32 {
            assert!((coeffs[i] - i as f32).abs() < 1e-6, "coeff {i}");
        }
    }

    #[test]
    fn seg_e2e_normalized_coords_are_letterbox_over_640() {
        // e2e segmenters (yolov10s / static fine-tunes) emit normalized
        // [0,1] xyxy relative to the 640x640 letterbox input, so decoding
        // is `v * 640 -> (v - pad) / scale`, not `v * orig_dim`.
        // rows must be > cols so this is recognized as "end-to-end"
        // (real e2e seg exports look like [300, 38]).
        let mut boxes = Array2::<f32>::zeros((40, 38));

        boxes[[0, 0]] = 0.25;
        boxes[[0, 1]] = 0.25;
        boxes[[0, 2]] = 0.65;
        boxes[[0, 3]] = 0.75;
        boxes[[0, 4]] = 0.9;
        boxes[[0, 5]] = 3.0;
        for i in 0..32 {
            boxes[[0, 6 + i]] = i as f32 * 0.5;
        }

        boxes[[1, 4]] = 0.02; // below threshold

        // 1280x720 frame letterboxed to 640x640: scale 0.5, pad_x 0, pad_y 140.
        let dets = parse_seg_output(boxes.view(), CONF, 1280, 720, 0.5, 0, 140)
            .expect("normalized e2e seg layout should parse");

        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(d.class_id, 3);
        // letterbox coords: [160, 160, 416, 480]
        assert!((d.x1 - 160.0 / 0.5).abs() < 1e-3); // 320
        assert!((d.y1 - (160.0 - 140.0) / 0.5).abs() < 1e-3); // 40
        assert!((d.x2 - 416.0 / 0.5).abs() < 1e-3); // 832
        assert!((d.y2 - (480.0 - 140.0) / 0.5).abs() < 1e-3); // 680
        let coeffs = d.mask_coefficients.expect("coefficients present");
        assert!((coeffs[7] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn seg_e2e_pixel_coords_unletterbox_like_raw_outputs() {
        // Real-world check with measured numbers from screenshot.png
        // (3840x2182): YOLO26n-seg-motion reports the person as
        // [313.9, 230.3, 399.4, 436.7] in letterbox-pixel space; the
        // letterbox-space COCO seg model reports
        // [313.7, 231.2, 400.5, 437.7] for the same person. The e2e pixel
        // box must un-letterbox to the same original-space box.
        let mut boxes = Array2::<f32>::zeros((40, 38));

        boxes[[0, 0]] = 313.9;
        boxes[[0, 1]] = 230.3;
        boxes[[0, 2]] = 399.4;
        boxes[[0, 3]] = 436.7;
        boxes[[0, 4]] = 0.97;
        boxes[[0, 5]] = 0.0;
        for i in 0..32 {
            boxes[[0, 6 + i]] = i as f32 * 0.25;
        }

        boxes[[1, 4]] = 0.40; // below threshold

        // 3840x2182: scale 1/6, pad_x 0, pad_y 138.
        let dets = parse_seg_output(boxes.view(), CONF, 3840, 2182, 1.0 / 6.0, 0, 138)
            .expect("pixel e2e seg layout should parse");

        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        // Expected original space: x [1883.4, 2396.4], y [553.8, 1792.2]
        assert!((d.x1 - 1883.4).abs() < 1.0, "x1={}", d.x1);
        assert!((d.y1 - 553.8).abs() < 1.0, "y1={}", d.y1);
        assert!((d.x2 - 2396.4).abs() < 1.0, "x2={}", d.x2);
        assert!((d.y2 - 1792.2).abs() < 1.0, "y2={}", d.y2);
        let coeffs = d.mask_coefficients.expect("coefficients present");
        assert!((coeffs[7] - 7 as f32 * 0.25).abs() < 1e-6);
    }

    /// Runs the real ONNX models (whatever is present in `models/`) through
    /// the exact app code path with a zero 640x640 input, verifying layout
    /// dispatch, shape handling and in-bounds outputs.
    #[test]
    fn smoke_real_models_on_zero_input() {
        let od_models = [
            "models/yolo26n.onnx",
            "models/yolov10s-instar-motion-19-750e.onnx",
        ];
        let seg_models = [
            "models/yolo26n-seg.onnx",
            "models/YOLO26n-seg-motion-600e-18062026.onnx",
        ];

        let zeros: Vec<f32> = vec![0.0; 3 * IMG as usize * IMG as usize];
        let zero_tensor = || {
            Tensor::from_shape(&[1usize, 3, 640, 640], &zeros).expect("zero tensor")
        };

        for path in od_models {
            if !Path::new(path).exists() {
                eprintln!("skipping missing OD model {path}");
                continue;
            }

            let model = load_model(path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"));
            let dets = run_od_inference(&model, zero_tensor(), CONF, IMG, IMG, 1.0, 0, 0)
                .unwrap_or_else(|e| panic!("OD inference failed on {path}: {e}"));

            assert!(dets.len() <= MAX_DETECTIONS, "{path}: {} detections", dets.len());
            for d in &dets {
                for v in [d.x1, d.y1, d.x2, d.y2] {
                    assert!((0.0..=640.0).contains(&v), "{path}: coordinate {v} out of bounds");
                }
            }

            eprintln!("{path}: {} detection(s)", dets.len());
        }

        for path in seg_models {
            if !Path::new(path).exists() {
                eprintln!("skipping missing SEG model {path}");
                continue;
            }

            let model = load_model(path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"));
            let seg = run_seg_inference(&model, zero_tensor(), CONF, IMG, IMG, 1.0, 0, 0)
                .unwrap_or_else(|e| panic!("SEG inference failed on {path}: {e}"));

            assert_eq!(seg.proto_masks.shape(), &[32, 160, 160]);
            assert!(
                seg.detections.len() <= MAX_DETECTIONS,
                "{path}: {} detections",
                seg.detections.len()
            );
            for d in &seg.detections {
                for v in [d.x1, d.y1, d.x2, d.y2] {
                    assert!((0.0..=640.0).contains(&v), "{path}: coordinate {v} out of bounds");
                }
                assert!(d.mask_coefficients.is_some(), "{path}: missing coefficients");
            }

            eprintln!("{path}: {} detection(s)", seg.detections.len());
        }
    }
}
