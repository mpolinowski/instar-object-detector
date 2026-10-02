// src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Models are loaded lazily (per run) from the paths selected in the UI,
// so nothing model-related is needed at startup.
fn main() {
    // Initialize video-rs decoder first to make sure FFMPEG dependency is operational
    video_rs::init().expect("FFmpeg/video-rs failed to initialize. Check system dependencies.");
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Info)
                .filter(|metadata| {
                    !metadata.target().starts_with("tract_core")
                        && !metadata.target().starts_with("tract_onnx")
                        && !metadata.target().starts_with("tract_linalg")
                })
                .build(),
        )
        // Provide the absolute path inside the handler so the macro can resolve the command metadata cross-crate
        .invoke_handler(tauri::generate_handler![tauri_yolo::run_inference])
        .run(tauri::generate_context!())
        .expect("error while running tauri");
}
