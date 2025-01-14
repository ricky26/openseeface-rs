use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use tracing::info;

use openseeface::tracker::{Tracker, TrackerConfig};

#[derive(Parser)]
struct Options {
    #[arg(long, help = "List available webcams")]
    pub list: bool,

    #[arg(short, long, help = "Camera index to use for tracking")]
    pub camera: Option<u32>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let opts = Options::parse();

    if opts.list {
        let cameras = nokhwa::query(ApiBackend::Auto)?;
        for camera in &cameras {
            info!("Camera {}: {}", camera.index(), camera.human_name());
        }
        return Ok(());
    }

    let camera_index = opts.camera.unwrap_or(0);
    let requested_format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(camera_index), requested_format)?;

    let config = TrackerConfig {
        ..Default::default()
    };

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    ctrlc::set_handler(move || {
        running_clone.store(false, Ordering::Relaxed);
    })?;

    info!(
        "Starting camera stream {:?}@{}",
        camera.resolution(),
        camera.frame_rate()
    );
    camera.open_stream()?;

    let mut tracker = Tracker::new(config)?;
    let epoch = Instant::now();
    while running.load(Ordering::Relaxed) {
        let now = Instant::now();
        let now64 = (now - epoch).as_secs_f64();

        let frame = camera.frame()?;
        let image = frame.decode_image::<RgbFormat>()?;

        info!("frame {:?}", frame.resolution());

        // ...
        tracker.detect(&image, now64)?;
    }

    camera.stop_stream()?;
    Ok(())
}
