use std::fmt::Write;
use std::time::Instant;

use bevy::asset::RenderAssetUsages;
use bevy::audio::AudioPlugin;
use bevy::color::palettes::css::{LIME, YELLOW};
use bevy::input::common_conditions::{input_just_pressed, input_toggle_active};
use bevy::log::Level;
use bevy::math::{uvec2, vec2, vec3};
use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::utils::tracing::span;
use bevy_inspector_egui::DefaultInspectorConfigPlugin;
use clap::Parser;
use image::buffer::ConvertBuffer;
use image::RgbaImage;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::{Buffer, CallbackCamera};

use openseeface::face::DEFAULT_FACE;
use openseeface::tracker::{Tracker, TrackerConfig, CONTOUR_INDICES};

pub mod inspector;

#[derive(Parser)]
struct Options {
    #[arg(long, help = "List available webcams")]
    pub list: bool,

    #[arg(short, long, help = "Camera index to use for tracking")]
    pub camera: Option<u32>,
}

fn main() -> anyhow::Result<()> {
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
    let mut camera = CallbackCamera::new(
        CameraIndex::Index(camera_index),
        requested_format,
        move |buffer| {
            let span = span!(Level::DEBUG, "camera frame");
            let _span = span.enter();
            frame_tx_clone.send(buffer).ok();
        },
    )?;

    let config = TrackerConfig {
        ..Default::default()
    };

    let camera_format = camera.camera_format()?;
    info!("Starting camera stream {}", camera_format);
    camera.open_stream()?;

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
            main_plugin,
        ));

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

    app
        .insert_resource(Epoch(epoch))
        .insert_resource(ActiveTracker(tracker))
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
            handle_new_camera_frames.run_if(input_toggle_active(true, KeyCode::F6)),
            draw_detections,
            draw_landmarks,
            save_obj.run_if(input_just_pressed(KeyCode::F10)),
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
pub struct ActiveTracker(Tracker);

#[derive(Clone, Debug, Resource)]
pub struct CameraImage(Handle<Image>);

#[derive(Clone, Debug, Resource)]
pub struct CameraMaterial(Handle<StandardMaterial>);

#[derive(Clone, Debug, Resource)]
pub struct CameraInfo {
    pub resolution: UVec2,
}

pub fn handle_new_camera_frames(
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera_image: Res<CameraImage>,
    camera_material: Res<CameraMaterial>,
    epoch: Res<Epoch>,
    frame_rx: Res<CameraFrameRx>,
    mut tracker: ResMut<ActiveTracker>,
) {
    if let Ok(frame) = frame_rx.0.try_recv() {
        let span = span!(Level::DEBUG, "frame");
        let _span = span.enter();

        let now = Instant::now();
        let now64 = (now - epoch.0).as_secs_f64();

        let image = span!(Level::DEBUG, "decode")
            .in_scope(|| frame.decode_image::<RgbFormat>())
            .expect("camera frame should be decodable");
        let image = image.convert();
        if let Err(err) = tracker.0.detect(&image, now64) {
            warn!("failed to track frame: {err}");
        }

        let span = span!(Level::DEBUG, "upload texture");
        let _span = span.enter();
        let Some(texture) = images.get_mut(&camera_image.0) else {
            return;
        };

        let rgba: RgbaImage = image.convert();
        texture.data.copy_from_slice(&rgba.as_raw());
        materials.get_mut(&camera_material.0);
    }
}

pub fn draw_detections(
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
    for &(min, size) in tracker.0.face_boxes() {
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

pub fn draw_landmarks(
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

    for face in tracker.0.faces() {
        for &(pos, c) in face.landmarks() {
            let p = transform_point(pos);
            let c = Color::hsv(c * 270., 1., 1.);
            gizmos.sphere(p, 0.01, c);
        }

        for &p in face.landmarks_camera() {
            let p = p.extend(-1.) * vec3(-0.4, 0.4, 1.);
            gizmos.sphere(p, 0.01, Color::BLACK);
        }

        if face.has_pose() {
            let r = face.rotation();
            let t = face.translation();

            for &p in &DEFAULT_FACE {
                let p = (r * p + t) * vec3(-0.4, 0.4, -1.0);
                gizmos.sphere(p, 0.01, YELLOW);
            }
        }
    }
}

pub fn save_obj(
    tracker: Res<ActiveTracker>,
) {
    let Some(face) = tracker.0.faces().first() else {
        warn!("No visible face, obj not saved");
        return;
    };

    let path = "face.obj";
    let mut contents = String::new();
    let landmarks = face.landmarks_camera();

    writeln!(&mut contents, "# Camera-space Landmarks").unwrap();

    for &p in landmarks {
        writeln!(&mut contents, "v {} {} 1", p.x, p.y).unwrap();
    }

    writeln!(&mut contents, "\n# Camera-space Landmarks\n#").unwrap();
    for &p in CONTOUR_INDICES.iter().filter_map(|&i| landmarks.get(i)) {
        writeln!(&mut contents, "# vec2({}, {})", p.x, p.y).unwrap();
    }

    if face.has_pose() {
        writeln!(&mut contents, "\n# Transformed Reference Mesh").unwrap();

        let r = face.rotation();
        let t = face.translation();

        for &p in &DEFAULT_FACE {
            let p = r * p + t;
            writeln!(&mut contents, "v {} {} {}", p.x, p.y, p.z).unwrap();
        }
    }

    if let Err(err) = std::fs::write(path, &contents) {
        error!("Failed to write face mesh: {err}");
    } else {
        info!("Face mesh saved to {path}");
    }
}
