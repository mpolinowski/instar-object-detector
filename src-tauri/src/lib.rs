pub mod modules {
    pub mod constants;
    pub mod drawing;
    pub mod image_utils;
    pub mod inference;
    pub mod models;
    pub mod video_utils;
}

pub use modules::inference::run_inference;
