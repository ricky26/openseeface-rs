use std::time::Instant;

use clap::Parser;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::CallbackCamera;
use image::buffer::ConvertBuffer;
use tracing::{info, span, Level};
use tracing_subscriber::{EnvFilter, Registry};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use openseeface::tracker::{Tracker, TrackerConfig};

#[derive(Parser)]
struct Options {
    #[arg(long, help = "List available webcams")]
    pub list: bool,

    #[arg(short, long, help = "Camera index to use for tracking")]
    pub camera: Option<u32>,
}

fn init_tracing() {
    let filter_layer = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();
    let subscriber = Registry::default().with(filter_layer);

    let fmt_layer = tracing_subscriber::fmt::layer();

    #[cfg(feature = "tracing")]
    let fmt_layer = tracing_subscriber::layer::Layer::with_filter(fmt_layer, tracing_subscriber::filter::filter_fn(|meta| {
        meta.fields().field("tracy.frame_mark").is_none()
    }));

    let subscriber = subscriber.with(fmt_layer);

    #[cfg(feature = "tracing")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default());

    subscriber.init();
}

fn main() -> anyhow::Result<()> {
    init_tracing();
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

    let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
    let frame_tx_clone = frame_tx.clone();
    let mut camera = CallbackCamera::new(CameraIndex::Index(camera_index), requested_format, move |buffer| {
        let span = span!(Level::DEBUG, "camera frame");
        let _span = span.enter();
        frame_tx_clone.send(Some(buffer)).ok();
    })?;

    let config = TrackerConfig {
        ..Default::default()
    };

    ctrlc::set_handler(move || {
        frame_tx.send(None).ok();
    })?;

    let camera_format = camera.camera_format()?;
    info!("Starting camera stream {}", camera_format);
    camera.open_stream()?;

    let mut tracker = Tracker::new(config)?;
    let epoch = Instant::now();
    while let Ok(Some(frame)) = frame_rx.recv() {
        let span = span!(Level::DEBUG, "frame");
        let _span = span.enter();

        let now = Instant::now();
        let now64 = (now - epoch).as_secs_f64();

        let image = span!(Level::DEBUG, "decode").in_scope(|| frame.decode_image::<RgbFormat>())?;
        let image = image.convert();
        tracker.detect(&image, now64)?;

        #[cfg(feature = "tracing")]
        tracing::event!(Level::DEBUG, message = "frame end", tracy.frame_mark = true);
    }

    camera.stop_stream()?;
    Ok(())
}
