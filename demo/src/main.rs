use std::fmt::Write;
use std::net::{SocketAddr, UdpSocket};
use std::time::Instant;

use anyhow::{anyhow, bail, Context};
use bevy::asset::RenderAssetUsages;
use bevy::audio::AudioPlugin;
use bevy::color::palettes::css::{DARK_BLUE, LIME, ORANGE, YELLOW};
use bevy::input::common_conditions::{input_just_pressed, input_toggle_active};
use bevy::log::Level;
use bevy::math::{uvec2, vec2, vec3};
use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::utils::tracing::span;
use bevy_inspector_egui::DefaultInspectorConfigPlugin;
use clap::Parser;
use image::{Rgb, Rgb32FImage, RgbImage, RgbaImage};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::{Buffer, CallbackCamera};

use openseeface::face::{DEFAULT_FACE, FACE_EDGES};
use openseeface::features::FeatureTracker;
use openseeface::features::openseeface::FeatureExtractor;
use openseeface::image::{rgb_to_rgb32f, rgb_to_rgba};
use openseeface::protocol::openseeface::FaceUpdate;
use openseeface::tracker::{Tracker, TrackerConfig, TrackerModel, CONTOUR_INDICES};

pub mod inspector;

pub mod features;

#[derive(Parser)]
struct Options {
    #[arg(short, long, help = "List available video sources")]
    pub list_cameras: bool,

    #[arg(short, long, help = "Camera index to use for tracking")]
    pub camera: Option<u32>,

    #[arg(short, long, help = "Camera resolution to use")]
    pub resolution: Option<String>,

    #[arg(short = 'm', long, help = "Maximum number of threads to use")]
    pub max_threads: Option<usize>,

    #[arg(short = 'f', long, default_value="1", help = "Maximum number of faces to detect")]
    pub max_faces: usize,

    #[arg(long, help = "Use RetinaFace for face detection")]
    pub use_retinaface: bool,

    #[arg(long, default_value = "3", help = "Pick face tracking model to use")]
    pub model: i32,

    #[arg(short, long, help = "Address to send OpenSeeFace packets to")]
    pub address: Option<String>,
}


fn main() -> anyhow::Result<()> {
    let opts = Options::parse();

    if opts.list_cameras {
        let cameras = nokhwa::query(ApiBackend::Auto)?;
        for camera in &cameras {
            info!("Camera {}: {}", camera.index(), camera.human_name());
        }
        return Ok(());
    }

    let camera_index = opts.camera.unwrap_or(0);
    let mut requested_format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    if let Some(s) = opts.resolution.as_ref() {
        let Some((ws, hs)) = s.split_once('x') else {
            bail!("Invalid resolution {s}")
        };

        let w = ws.parse()?;
        let h = hs.parse()?;
        let resolution = Resolution::new(w, h);
        requested_format = RequestedFormat::new::<RgbFormat>(
            RequestedFormatType::HighestResolution(resolution));
    }

    let (frame_tx, frame_rx) = crossbeam_channel::bounded(0);
    let frame_tx_clone = frame_tx.clone();
    let mut camera = CallbackCamera::new(
        CameraIndex::Index(camera_index),
        requested_format,
        move |buffer| {
            let span = span!(Level::DEBUG, "camera frame");
            let _span = span.enter();
            frame_tx_clone.send(buffer).ok();
        },
    )?;

    let mut config = TrackerConfig {
        model_type: TrackerModel::from_i32(opts.model)
            .ok_or_else(|| anyhow!("invalid model type '{}'", opts.model))?,
        max_faces: opts.max_faces,
        use_retinaface: opts.use_retinaface,
        ..Default::default()
    };

    if let Some(max_threads) = opts.max_threads {
        config.max_threads = max_threads;
    }

    let features = FeatureTracker::new(config.max_faces);
    let tracker = Tracker::new(config)?;
    let epoch = Instant::now();
    let mut app = App::default();
    app
        .add_plugins((
            DefaultPlugins.build()
                .disable::<AudioPlugin>(),
            DefaultInspectorConfigPlugin,
            bevy_egui::EguiPlugin,
            inspector::plugin,
            features::plugin,
            main_plugin,
        ));

    let camera_format = camera.camera_format()?;
    info!("Starting camera stream {}", camera_format);
    camera.open_stream()?;

    if let Some(target) = opts.address.as_ref() {
        let target = target.parse()
            .context("parsing target address")?;
        let socket = UdpSocket::bind("127.0.0.1:0")?;
        app.insert_resource(OsfTarget {
            socket,
            target,
        });
    }

    let camera_image = Image::new_fill(
        Extent3d {
            width: camera_format.width(),
            height: camera_format.height(),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );

    let camera_image = app.world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_image);

    let active_tracker = ActiveTracker {
        tracker,
        features,
    };

    app
        .init_resource::<CameraImageBuffers>()
        .insert_resource(Epoch(epoch))
        .insert_resource(active_tracker)
        .insert_resource(CameraFrameRx(frame_rx))
        .insert_resource(CameraImage(camera_image))
        .insert_resource(CameraInfo {
            resolution: uvec2(camera_format.width(), camera_format.height()),
        })
        .run();

    camera.stop_stream()?;
    Ok(())
}

fn main_plugin(app: &mut App) {
    app
        .add_systems(Startup, setup)
        .add_systems(Update, (
            (
                handle_new_camera_frames.run_if(input_toggle_active(true, KeyCode::F6)),
                (
                    (
                        detect_faces,
                        upload_camera_image,
                    ).run_if(new_camera_image),
                    draw_detections.run_if(input_toggle_active(false, KeyCode::F1)),
                    draw_landmarks.run_if(input_toggle_active(false, KeyCode::F2)),
                    draw_reference_face.run_if(input_toggle_active(false, KeyCode::F3)),
                    draw_face_3d.run_if(input_toggle_active(true, KeyCode::F4)),
                ),
            ).chain(),
            send_packets
                .after(detect_faces)
                .run_if(resource_exists::<OsfTarget>)
                .run_if(new_camera_image),
            save_obj.run_if(input_just_pressed(KeyCode::F10)),
            dump_debug_images.run_if(input_just_pressed(KeyCode::F11)),
        ));
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera_image: Res<CameraImage>,
    camera_info: Res<CameraInfo>,
) {
    let plane_size = 2.0;
    let plane_aspect = (camera_info.resolution.x as f32) / (camera_info.resolution.y as f32);
    let plane_half_size = vec2(plane_size * plane_aspect, plane_size);
    let mut plane_mesh = Plane3d::new(Vec3::Z, plane_half_size)
        .mesh()
        .build();
    if let Some(VertexAttributeValues::Float32x2(uvs)) = plane_mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0) {
        for uv in uvs {
            uv[0] = 1. - uv[0];
        }
    }
    let plane = meshes.add(plane_mesh);
    let camera_material = materials.add(StandardMaterial {
        unlit: true,
        base_color_texture: Some(camera_image.0.clone()),
        ..default()
    });

    commands.insert_resource(CameraMaterial(camera_material.clone()));

    commands.spawn((
        Name::new("Camera Plane"),
        Mesh3d(plane),
        MeshMaterial3d(camera_material),
        Transform::from_translation(vec3(0., 0., -5.)),
    ));

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
    ));
}

#[derive(Clone, Debug, Resource)]
pub struct Epoch(Instant);

#[derive(Clone, Debug, Resource)]
pub struct CameraFrameRx(crossbeam_channel::Receiver<Buffer>);

#[derive(Resource)]
pub struct ActiveTracker {
    pub tracker: Tracker,
    pub features: FeatureTracker<FeatureExtractor>,
}

#[derive(Clone, Debug, Resource)]
pub struct CameraImage(Handle<Image>);

#[derive(Clone, Debug, Resource)]
pub struct CameraMaterial(Handle<StandardMaterial>);

#[derive(Clone, Debug, Resource)]
pub struct CameraInfo {
    pub resolution: UVec2,
}

#[derive(Clone, Debug, Default, Resource)]
pub struct CameraImageBuffers {
    pub rgb: RgbImage,
    pub float: Rgb32FImage,
}

fn new_camera_image(buffers: Res<CameraImageBuffers>) -> bool {
    buffers.is_changed() && !buffers.rgb.as_raw().is_empty()
}


fn handle_new_camera_frames(
    frame_rx: Res<CameraFrameRx>,
    mut camera_images: ResMut<CameraImageBuffers>,
) {
    if let Ok(frame) = frame_rx.0.try_recv() {
        let camera_images = &mut *camera_images;
        camera_images.rgb = span!(Level::TRACE, "decode")
            .in_scope(|| frame.decode_image::<RgbFormat>())
            .expect("camera frame should be decodable");
        span!(Level::TRACE, "convert")
            .in_scope(|| rgb_to_rgb32f(&mut camera_images.float, &camera_images.rgb));
    }
}

fn detect_faces(
    epoch: Res<Epoch>,
    mut tracker: ResMut<ActiveTracker>,
    camera_images: Res<CameraImageBuffers>,
) {
    let now = Instant::now();
    let now64 = (now - epoch.0).as_secs_f64();

    let tracker = &mut *tracker;
    if let Err(err) = tracker.tracker.detect(&camera_images.float) {
        warn!("failed to track frame: {err}");
    }

    let faces = tracker.tracker.faces();
    tracker.features.update(faces, now64);
}

fn upload_camera_image(
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera_image: Res<CameraImage>,
    camera_material: Res<CameraMaterial>,
    camera_images: Res<CameraImageBuffers>,
) {
    let Some(texture) = images.get_mut(&camera_image.0) else {
        return;
    };

    let mut rgba = RgbaImage::from_raw(
        texture.width(), texture.height(), std::mem::take(&mut texture.data)).unwrap();
    rgb_to_rgba(&mut rgba, &camera_images.rgb);
    texture.data = rgba.into_raw();
    materials.get_mut(&camera_material.0);
}

fn draw_detections(
    mut gizmos: Gizmos,
    camera_info: Res<CameraInfo>,
    tracker: Res<ActiveTracker>,
) {
    let z = -4.9;
    let point_scale = -4. / (camera_info.resolution.y as f32);
    let point_offset = -camera_info.resolution.as_vec2() * 0.5 * point_scale;

    let transform_point = move |pt: Vec2| {
        ((pt * point_scale) + point_offset).extend(z)
    };

    let color = LIME;
    for &(min, size) in tracker.tracker.face_detections() {
        let max = min + size;
        let a = transform_point(vec2(min.x, min.y));
        let b = transform_point(vec2(max.x, min.y));
        let c = transform_point(vec2(max.x, max.y));
        let d = transform_point(vec2(min.x, max.y));

        gizmos.line(a, b, color);
        gizmos.line(b, c, color);
        gizmos.line(c, d, color);
        gizmos.line(d, a, color);
    }
}

fn draw_landmarks(
    mut gizmos: Gizmos,
    tracker: Res<ActiveTracker>,
) {
    let landmarks_z = -4.5;
    let camera_z = -1.25;
    let scale = (landmarks_z / camera_z) * vec3(-2., 2., 1.);

    for face in tracker.tracker.visible_faces() {
        let landmarks_camera = face.landmarks_camera();
        for (&p, &c) in landmarks_camera.iter().zip(face.landmark_confidence()) {
            let p = p.extend(camera_z) * scale;
            let c = Color::hsv(c * 270., 1., 1.);
            gizmos.sphere(p, 0.01, c);
        }

        for &(a, b) in &FACE_EDGES[..(FACE_EDGES.len() - 2)] {
            let pa = landmarks_camera[a].extend(camera_z) * scale;
            let pb = landmarks_camera[b].extend(camera_z) * scale;
            gizmos.line(pa, pb, DARK_BLUE);
        }
    }
}

fn draw_reference_face(
    mut gizmos: Gizmos,
    tracker: Res<ActiveTracker>,
) {
    for face in tracker.tracker.visible_faces() {
        if !face.has_pose() {
            continue;
        }

        let r = face.rotation();
        let t = face.translation();

        let transform_point = move |p|
            (r * p + t) * vec3(-1., 1., -0.6);

        for &p in &DEFAULT_FACE {
            let p = transform_point(p);
            gizmos.sphere(p, 0.01, YELLOW);
        }

        for &(a, b) in &FACE_EDGES {
            let pa = transform_point(DEFAULT_FACE[a]);
            let pb = transform_point(DEFAULT_FACE[b]);
            gizmos.line(pa, pb, YELLOW);
        }
    }
}

fn draw_face_3d(
    mut gizmos: Gizmos,
    tracker: Res<ActiveTracker>,
) {
    for face in tracker.tracker.visible_faces() {
        if !face.has_pose() {
            continue;
        }

        let face_3d = face.face_3d();
        let r = face.rotation();
        let t = face.translation();
        let transform_point = move |p|
            (r * p + t) * vec3(-1.6, 1.6, -1.) * 0.5;

        for (&p, &c) in face_3d.iter().zip(face.landmark_confidence()) {
            let p = transform_point(p);
            let c = Color::hsv(c * 270., 1., 1.);
            gizmos.sphere(p, 0.01, c);
        }

        for &(a, b) in &FACE_EDGES {
            let pa = transform_point(face_3d[a]);
            let pb = transform_point(face_3d[b]);
            gizmos.line(pa, pb, ORANGE);
        }
    }
}

fn save_obj(
    tracker: Res<ActiveTracker>,
) {
    let Some(face) = tracker.tracker.visible_faces().last() else {
        warn!("No visible face, obj not saved");
        return;
    };

    let path = "face.obj";
    let mut contents = String::new();
    let landmarks = face.landmarks_camera();
    let mut v = 0;

    writeln!(&mut contents, "# Camera-space Landmarks\no landmarks").unwrap();

    let vo = v;
    for &p in landmarks {
        writeln!(&mut contents, "v {} {} 1", p.x, p.y).unwrap();
        v += 1;
    }
    for &(a, b) in &FACE_EDGES {
        writeln!(&mut contents, "l {} {}", a + vo + 1, b + vo + 1).unwrap();
    }

    writeln!(&mut contents, "\n# Contour\n#").unwrap();
    for &p in CONTOUR_INDICES.iter().filter_map(|&i| landmarks.get(i)) {
        writeln!(&mut contents, "# vec2({}, {}),", p.x, p.y).unwrap();
    }

    if face.has_pose() {
        writeln!(&mut contents, "\n# Reference Mesh\no reference").unwrap();

        let vo = v;
        for &p in &DEFAULT_FACE {
            writeln!(&mut contents, "v {} {} {}", p.x, p.y, p.z).unwrap();
            v += 1;
        }
        for &(a, b) in &FACE_EDGES {
            writeln!(&mut contents, "l {} {}", a + vo + 1, b + vo + 1).unwrap();
        }

        writeln!(&mut contents, "\n# Transformed Reference Mesh\no transformed_reference").unwrap();

        let r = face.rotation();
        let t = face.translation();

        let vo = v;
        for &p in &DEFAULT_FACE {
            let p = r * p + t;
            writeln!(&mut contents, "v {} {} {}", p.x, p.y, p.z).unwrap();
            v += 1;
        }
        for &(a, b) in &FACE_EDGES {
            writeln!(&mut contents, "l {} {}", a + vo + 1, b + vo + 1).unwrap();
        }

        writeln!(&mut contents, "\n# 3D Face\no face_3d").unwrap();

        let vo = v;
        for &p in face.face_3d() {
            writeln!(&mut contents, "v {} {} {}", p.x, p.y, p.z).unwrap();
            v += 1;
        }
        for &(a, b) in &FACE_EDGES {
            writeln!(&mut contents, "l {} {}", a + vo + 1, b + vo + 1).unwrap();
        }

        writeln!(&mut contents, "\n# Transformed 3D Face\no transformed_face_3d").unwrap();

        let vo = v;
        for &p in face.face_3d() {
            let p = r * p + t;
            writeln!(&mut contents, "v {} {} {}", p.x, p.y, p.z).unwrap();
            v += 1;
        }
        for &(a, b) in &FACE_EDGES {
            writeln!(&mut contents, "l {} {}", a + vo + 1, b + vo + 1).unwrap();
        }
    }

    if let Err(err) = std::fs::write(path, &contents) {
        error!("Failed to write face mesh: {err}");
    } else {
        info!("Face mesh saved to {path}");
    }
}

#[derive(Resource)]
pub struct OsfTarget {
    socket: UdpSocket,
    target: SocketAddr,
}

fn send_packets(
    time: Res<Time>,
    target: Res<OsfTarget>,
    camera: Res<CameraInfo>,
    tracker: Res<ActiveTracker>,
    mut buffer: Local<Vec<u8>>,
) {
    if tracker.tracker.num_visible_faces() == 0 {
        return;
    }

    buffer.clear();

    for (index, face) in tracker.tracker.faces().iter().enumerate() {
        if !face.is_alive() {
            continue;
        }

        let features = &tracker.features.current_features()[index];
        let update = FaceUpdate::from_tracked_face(
            face,
            features,
            camera.resolution.x as f32,
            camera.resolution.y as f32,
            time.elapsed_secs_f64(),
        );
        update.write::<byteorder::LittleEndian>(&mut *buffer);
    }

    target.socket.send_to(&buffer, target.target).unwrap();
}

fn dump_debug_images(
    tracker: Res<ActiveTracker>,
) {
    for (name, image) in tracker.tracker.iter_debug_images() {
        let path = format!("{name}.exr");
        let mut image = tracker.tracker.denormalise_image(image);
        if let Err(err) = image.save(&path) {
            error!("Failed to write {path}: {err}");
        } else {
            info!("Wrote {path}");
        }

        if name == "face_scratch_res" {
            if let Some(face) = tracker.tracker.visible_faces().last() {
                let path = "landmarks.exr";
                let out_size = uvec2(image.width(), image.height()).as_vec2();
                let min = face.image_min().as_vec2();
                let max = face.image_max().as_vec2();
                let src_size = max - min;
                let scale = out_size / src_size;
                let rgb = Rgb([0.0, 0.0, 1.0]);
                for &p in face.landmarks_image() {
                    let p2 = (p - min) * scale;
                    let up = p2.as_uvec2();
                    image.put_pixel(up.x, up.y, rgb);
                }

                if let Err(err) = image.save(path) {
                    error!("Failed to write {path}: {err}");
                } else {
                    info!("Wrote {path}");
                }
            }
        }
    }
}
