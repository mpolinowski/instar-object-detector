use crate::modules::models::PreprocessedImage;
use anyhow::Context;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use std::path::Path;
use tract_onnx::prelude::*;

pub fn is_image(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    matches!(ext.as_str(), "jpg" | "jpeg" | "png")
}

pub fn load_and_preprocess_image(path: &Path, img_size: usize) -> TractResult<PreprocessedImage> {
    let original_img =
        image::open(path).with_context(|| format!("Failed to open image at {:?}", path))?;
    let (orig_w, orig_h) = original_img.dimensions();

    let (letterboxed, scale, pad_x, pad_y) = letterbox(&original_img, img_size as u32);

    let input_tensor: Tensor =
        tract_ndarray::Array4::from_shape_fn((1, 3, img_size, img_size), |(_, c, y, x)| {
            let pixel = letterboxed.get_pixel(x as u32, y as u32);
            pixel[c] as f32 / 255.0
        })
        .into();

    Ok(PreprocessedImage {
        original_img,
        input_tensor,
        orig_w,
        orig_h,
        scale,
        pad_x,
        pad_y,
    })
}

pub fn letterbox(
    img: &DynamicImage,
    new_size: u32,
) -> (ImageBuffer<Rgb<u8>, Vec<u8>>, f32, u32, u32) {
    let (orig_w, orig_h) = img.dimensions();

    let scale = f32::min(
        new_size as f32 / orig_w as f32,
        new_size as f32 / orig_h as f32,
    );

    let resized_w = (orig_w as f32 * scale).round() as u32;
    let resized_h = (orig_h as f32 * scale).round() as u32;

    let resized_rgba = image::imageops::resize(
        img,
        resized_w,
        resized_h,
        image::imageops::FilterType::Triangle,
    );

    let resized = DynamicImage::ImageRgba8(resized_rgba).to_rgb8();
    let mut canvas = ImageBuffer::from_pixel(new_size, new_size, Rgb([114, 114, 114]));

    let pad_x = (new_size - resized_w) / 2;
    let pad_y = (new_size - resized_h) / 2;

    image::imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    (canvas, scale, pad_x, pad_y)
}
