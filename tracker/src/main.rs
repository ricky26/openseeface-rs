use std::net::{SocketAddr, UdpSocket};
use std::time::Instant;

use anyhow::anyhow;
use clap::Parser;
use glam::{vec3, EulerRot};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::CallbackCamera;
use image::buffer::ConvertBuffer;
use tracing::{info, span, Level};
use tracing_subscriber::{EnvFilter, Registry};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use openseeface::protocol::FaceUpdate;
use openseeface::tracker::{Tracker, TrackerConfig, TrackerModel};

#[derive(Parser)]
struct Options {
    #[arg(short, long, help = "List available video sources")]
    pub list_cameras: bool,

    #[arg(short, long, help = "Camera index to use for tracking")]
    pub camera: Option<u32>,

    #[arg(short = 'm', long, help = "Maximum number of threads to use")]
    pub max_threads: Option<usize>,

    #[arg(short = 'f', long, default_value="1", help = "Maximum number of faces to detect")]
    pub max_faces: usize,

    #[arg(long, help = "Use RetinaFace for face detection")]
    pub use_retinaface: bool,

    #[arg(long, default_value = "3", help = "Pick face tracking model to use")]
    pub model: i32,

    #[arg(short, long, default_value = "127.0.0.1:11573", help = "Address to send OpenSeeFace packets to")]
    pub address: String,
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

#[tracing::instrument(skip_all)]
fn send_packet(
    time: f64,
    width: f32,
    height: f32,
    tracker: &Tracker,
    socket: &UdpSocket,
    target: &SocketAddr,
    buffer: &mut Vec<u8>,
) -> std::io::Result<()> {
    if tracker.faces().is_empty() {
        return Ok(());
    }

    buffer.clear();

    for face in tracker.faces() {
        let update = FaceUpdate::from_tracked_face(face, time);
        update.write::<byteorder::LittleEndian>(&mut *buffer);
    }

    socket.send_to(buffer, target)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let opts = Options::parse();

    if opts.list_cameras {
        let cameras = nokhwa::query(ApiBackend::Auto)?;
        for camera in &cameras {
            info!("Camera {}: {}", camera.index(), camera.human_name());
        }
        return Ok(());
    }

    let target = opts.address.parse()?;

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

    let mut config = TrackerConfig {
        use_retinaface: opts.use_retinaface,
        model_type: TrackerModel::from_i32(opts.model)
            .ok_or_else(|| anyhow!("invalid model type '{}'", opts.model))?,
        ..Default::default()
    };

    if let Some(max_threads) = opts.max_threads {
        config.max_threads = max_threads;
    }

    ctrlc::set_handler(move || {
        frame_tx.send(None).ok();
    })?;

    let socket = UdpSocket::bind("127.0.0.1:0")?;
    info!("Sending OSF protocol to {}", &opts.address);

    let camera_format = camera.camera_format()?;
    info!("Starting camera stream {}", camera_format);
    camera.open_stream()?;

    let mut tracker = Tracker::new(config)?;
    let epoch = Instant::now();
    let width = camera_format.width() as f32;
    let height = camera_format.height() as f32;
    let mut buffer = Vec::new();
    while let Ok(Some(frame)) = frame_rx.recv() {
        let span = span!(Level::DEBUG, "frame");
        let _span = span.enter();

        let now = Instant::now();
        let now64 = (now - epoch).as_secs_f64();

        let image = span!(Level::DEBUG, "decode").in_scope(|| frame.decode_image::<RgbFormat>())?;
        let image = span!(Level::DEBUG, "convert").in_scope(|| image.convert());
        tracker.detect(&image, now64)?;

        send_packet(now64, width, height, &tracker, &socket, &target, &mut buffer)?;

        #[cfg(feature = "tracing")]
        tracing::event!(Level::DEBUG, message = "frame end", tracy.frame_mark = true);
    }

    camera.stop_stream()?;
    Ok(())
}
