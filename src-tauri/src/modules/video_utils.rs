use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use image::{GrayImage, Luma};
use imageproc::contours::find_contours;
use imageproc::distance_transform::Norm;
use imageproc::morphology::dilate;
use std::path::Path;
use video_rs::decode::Decoder;

// Explicitly bring tract's Tensor type into scope
use tract_onnx::prelude::{tract_ndarray, Tensor};

use crate::modules::image_utils::letterbox;
use crate::modules::models::{BBox, PreprocessedImage};

pub fn preprocess_frame(frame: &DynamicImage, img_size: usize) -> PreprocessedImage {
    let original_img = frame.clone();
    let (orig_w, orig_h) = original_img.dimensions();
    let (letterboxed, scale, pad_x, pad_y) = letterbox(&original_img, img_size as u32);

    // Using tract's internal ndarray array format mapping
    let input_tensor: Tensor =
        tract_ndarray::Array4::from_shape_fn((1, 3, img_size, img_size), |(_, c, y, x)| {
            let pixel = letterboxed.get_pixel(x as u32, y as u32);
            pixel[c] as f32 / 255.0
        })
        .into();

    PreprocessedImage {
        original_img,
        input_tensor,
        orig_w,
        orig_h,
        scale,
        pad_x,
        pad_y,
    }
}

pub fn is_video(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(ext.as_str(), "avi" | "mp4" | "mkv" | "mov")
}

// Finds the first frame with notable motion.
// Returns the detected motion bounding boxes and the DynamicImage (RGB) for inference/drawing pipelines.
pub fn find_motion_pixel(
    video_path: &Path,
    start_frame: usize,
    tries: isize,
    interval: usize,
    threshold_val: u8,
) -> Option<(Vec<BBox>, DynamicImage)> {
    // Note: video_rs::init() is now called in main.rs

    let mut decoder = match Decoder::new(video_path) {
        Ok(d) => d,
        Err(e) => {
            println!("ERROR: Failed to open video file {:?}: {:?}", video_path, e);
            return None;
        }
    };

    let mut runs = 0;
    let mut static_background: Option<GrayImage> = None;

    // Determine total tries
    let max_tries = if tries == -1 { 500 } else { tries as usize };

    while runs < max_tries {
        // Sequentially grab the frame data.
        // For the very first iteration, we skip to start_frame.
        let frame_result = if runs == 0 && start_frame > 0 {
            decoder.decode_iter().nth(start_frame)
        } else if interval > 1 {
            decoder.decode_iter().nth(interval - 1)
        } else {
            decoder.decode_iter().next()
        };

        if let Some(Ok((_, ndarray_frame))) = frame_result {
            runs += 1;
            let height = ndarray_frame.shape()[0] as u32;
            let width = ndarray_frame.shape()[1] as u32;

            let mut gray_img = GrayImage::new(width, height);
            let mut rgb_buffer = ImageBuffer::new(width, height);

            // Convert ndarray to Image types
            for y in 0..height {
                for x in 0..width {
                    let r = ndarray_frame[[y as usize, x as usize, 0]];
                    let g = ndarray_frame[[y as usize, x as usize, 1]];
                    let b = ndarray_frame[[y as usize, x as usize, 2]];
                    rgb_buffer.put_pixel(x, y, Rgb([r, g, b]));

                    let gray_val = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                    gray_img.put_pixel(x, y, Luma([gray_val]));
                }
            }

            let blurred = imageproc::filter::gaussian_blur_f32(&gray_img, 21.0);
            let background = static_background.get_or_insert_with(|| blurred.clone());

            // Compute Difference & Threshold
            let mut thresh_frame = GrayImage::new(width, height);
            for (x, y, p_bg) in background.enumerate_pixels() {
                let p_diff = blurred.get_pixel(x, y);
                let diff = (p_bg.0[0] as i16 - p_diff.0[0] as i16).abs() as u8;
                let val = if diff > threshold_val { 255 } else { 0 };
                thresh_frame.put_pixel(x, y, Luma([val]));
            }

            let dilated_frame = dilate(&thresh_frame, Norm::LInf, 2);
            let contours = find_contours::<u32>(&dilated_frame);

            let mut bboxes = Vec::new();
            for contour in contours {
                if contour.points.is_empty() {
                    continue;
                }
                let (mut min_x, mut max_x) = (u32::MAX, u32::MIN);
                let (mut min_y, mut max_y) = (u32::MAX, u32::MIN);

                for pt in &contour.points {
                    min_x = min_x.min(pt.x);
                    max_x = max_x.max(pt.x);
                    min_y = min_y.min(pt.y);
                    max_y = max_y.max(pt.y);
                }

                if (max_x - min_x) > 10 && (max_y - min_y) > 10 {
                    bboxes.push(BBox {
                        x: min_x,
                        y: min_y,
                        width: max_x - min_x,
                        height: max_y - min_y,
                    });
                }
            }

            if !bboxes.is_empty() {
                return Some((bboxes, DynamicImage::ImageRgb8(rgb_buffer)));
            }
        } else {
            break; // End of video
        }
    }
    None
}
