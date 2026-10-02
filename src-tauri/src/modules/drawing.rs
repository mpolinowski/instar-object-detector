use crate::modules::models::{BBox, Detection};

use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use once_cell::sync::Lazy;
use tract_onnx::prelude::tract_ndarray::Array3;

/// Resolved label table: entry `i` is the (name, RGB color) for model class `i`.
/// Built by inference from either a built-in table (COCO / Motion / Static) or
/// a user-supplied label file, so the same drawing code works for any model.
pub type LabelSet = Vec<(String, [u8; 3])>;

pub fn include_font() -> &'static FontArc {
    static FONT: Lazy<FontArc> = Lazy::new(|| {
        let font_data: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

        FontArc::try_from_slice(font_data).unwrap()
    });

    &FONT
}

pub fn draw_detection(
    canvas: &mut RgbImage,
    det: &Detection,
    font: &FontArc,
    labels: &LabelSet,
) {
    // Out-of-range class ids (e.g. a model with more classes than the label
    // table) fall back to a neutral box instead of failing.
    let fallback = format!("class {}", det.class_id);
    let (class_name, raw_color) = match labels.get(det.class_id) {
        Some((name, color)) => (name.as_str(), *color),
        None => (fallback.as_str(), [128, 128, 128]),
    };

    let color = Rgb(raw_color);

    let left = det.x1.min(det.x2).round() as i32;
    let top = det.y1.min(det.y2).round() as i32;
    let right = det.x1.max(det.x2).round() as i32;
    let bottom = det.y1.max(det.y2).round() as i32;

    if right <= left || bottom <= top {
        return;
    }

    let rect = Rect::at(left, top).of_size((right - left) as u32, (bottom - top) as u32);

    draw_hollow_rect_mut(canvas, rect, color);

    draw_text_mut(
        canvas,
        Rgb([255, 255, 255]),
        left,
        (top - 20).max(0),
        PxScale::from(20.0),
        font,
        &format!("{} {:.2}", class_name, det.score),
    );
}

pub fn draw_detection_with_mask(
    canvas: &mut RgbImage,
    det: &Detection,
    font: &FontArc,
    proto_masks: &Array3<f32>,
    scale: f32,
    pad_x: u32,
    pad_y: u32,
    labels: &LabelSet,
) {
    draw_detection(canvas, det, font, labels);

    if let Some(coeffs) = &det.mask_coefficients {
        let mut instance_mask = [[0.0f32; 160]; 160];

        for y in 0..160 {
            for x in 0..160 {
                let mut sum = 0.0;

                for i in 0..32 {
                    sum += coeffs[i] * proto_masks[[i, y, x]];
                }

                instance_mask[y][x] = 1.0 / (1.0 + (-sum).exp());
            }
        }

        let x1_img = det.x1;
        let y1_img = det.y1;
        let x2_img = det.x2;
        let y2_img = det.y2;

        let width = (x2_img - x1_img).max(1.0) as u32;
        let height = (y2_img - y1_img).max(1.0) as u32;

        for mx in 0..width {
            for my in 0..height {
                // map original image pixel back to the 640x640 input space
                let ix = (x1_img + mx as f32) * scale + pad_x as f32;
                let iy = (y1_img + my as f32) * scale + pad_y as f32;

                // map that 640x640 coordinate to the 160x160 prototype coordinate
                let px = (ix / 4.0).round() as usize;
                let py = (iy / 4.0).round() as usize;

                if px < 160 && py < 160 {
                    let val = instance_mask[py][px];
                    if val > 0.5 {
                        let cx = (x1_img as i32) + mx as i32;
                        let cy = (y1_img as i32) + my as i32;

                        if cx >= 0
                            && cy >= 0
                            && cx < canvas.width() as i32
                            && cy < canvas.height() as i32
                        {
                            let pixel = canvas.get_pixel_mut(cx as u32, cy as u32);
                            pixel.0[0] = ((pixel.0[0] as f32 * 0.6) + 255.0 * 0.4) as u8;
                        }
                    }
                }
            }
        }
    }
}

pub fn draw_motion_overlays(render_target: &mut RgbaImage, motion_boxes: &[BBox]) {
    let fill = Rgba([0, 200, 200, 64]);

    let alpha = fill.0[3] as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;

    for bbox in motion_boxes {
        let x1 = bbox.x;
        let y1 = bbox.y;

        let x2 = (bbox.x + bbox.width).min(render_target.width());
        let y2 = (bbox.y + bbox.height).min(render_target.height());

        // bounds safety guard to prevent out-of-bounds loops
        if x1 >= render_target.width() || y1 >= render_target.height() {
            continue;
        }

        for y in y1..y2 {
            for x in x1..x2 {
                let p = render_target.get_pixel_mut(x, y);

                p.0[0] = (fill.0[0] as f32 * alpha + p.0[0] as f32 * inv_alpha) as u8;
                p.0[1] = (fill.0[1] as f32 * alpha + p.0[1] as f32 * inv_alpha) as u8;
                p.0[2] = (fill.0[2] as f32 * alpha + p.0[2] as f32 * inv_alpha) as u8;
            }
        }
    }
}
