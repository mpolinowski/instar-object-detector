use image::DynamicImage;
use tract_onnx::prelude::tract_ndarray::Array3;
use tract_onnx::prelude::Tensor;

#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
    pub mask_coefficients: Option<[f32; 32]>,
}

pub struct PreprocessedImage {
    pub original_img: DynamicImage,
    pub input_tensor: Tensor,
    pub orig_w: u32,
    pub orig_h: u32,
    pub scale: f32,
    pub pad_x: u32,
    pub pad_y: u32,
}

#[derive(Debug, Clone)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct SegInferenceResult {
    pub detections: Vec<Detection>,
    pub proto_masks: Array3<f32>, // Shape: [32, 160, 160]
}
