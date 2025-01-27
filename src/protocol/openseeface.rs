//! Implementation of OpenSeeFace protocol.
//!
//! This protocol is used to communicate current pose information to VSeeFace
//! and other OpenSeeFace compatible software.

use std::io::Write;

use byteorder::{ByteOrder, WriteBytesExt};
use glam::{vec3, EulerRot, Quat, Vec2, Vec3};

use crate::face::TrackedFace;
use crate::features::openseeface::Features;

/// A single face update packet.
///
/// These are used to broadcast new face pose information.
pub struct FaceUpdate<'a> {
    /// Time this update was recorded.
    ///
    /// This can be from any offset but should be counted in seconds.
    pub timestamp: f64,
    /// The ID of this face.
    ///
    /// Each currently visible face should have a unique ID, and IDs should
    /// persist whilst the face remains visible.
    pub face_id: u32,
    /// Width of the source image.
    pub width: f32,
    /// Height of the source image.
    pub height: f32,
    /// True if the face was detected in the image.
    ///
    /// An update with success set to false set to false should be sent when
    /// tracking of a given face is lost.
    pub success: bool,
    /// The squared error of the perspective-n-point placement of the reference
    /// face.
    pub pnp_error: f32,
    /// Blink amount of eye leftmost in the image, 0..1.
    pub blink_left: f32,
    /// Blink amount of eye rightmost in the image, 0..1.
    pub blink_right: f32,
    /// Rotation of the face in world space.
    pub rotation: Quat,
    /// [`Self::rotation`] as an Euler rotation.
    pub rotation_euler: Vec3,
    /// Translation of the face in world sapce.
    pub translation: Vec3,
    /// Confidence values for each landmark.
    pub landmark_confidence: &'a [f32],
    /// 2D image-space coordinates for each landmark.
    pub landmarks: &'a [Vec2],
    /// 3D world-space coordinates for each landmark.
    pub landmarks_3d: &'a [Vec3],
    /// OpenSeeFace features detected for this face.
    pub features: &'a Features,
}

impl FaceUpdate<'_> {
    /// Write this update to a buffer.
    ///
    /// Current implementations always use little-endian.
    pub fn write<E: ByteOrder>(&self, out: &mut impl Write) {
        out.write_f64::<E>(self.timestamp).unwrap();
        out.write_u32::<E>(self.face_id).unwrap();
        out.write_f32::<E>(self.width).unwrap();
        out.write_f32::<E>(self.height).unwrap();
        out.write_f32::<E>(self.blink_right).unwrap();
        out.write_f32::<E>(self.blink_left).unwrap();
        out.write_u8(if self.success { 1 } else { 0 }).unwrap();
        out.write_f32::<E>(self.pnp_error).unwrap();
        out.write_f32::<E>(-self.rotation.y).unwrap();
        out.write_f32::<E>(self.rotation.x).unwrap();
        out.write_f32::<E>(-self.rotation.z).unwrap();
        out.write_f32::<E>(self.rotation.w).unwrap();
        out.write_f32::<E>(self.rotation_euler.x).unwrap();
        out.write_f32::<E>(self.rotation_euler.y).unwrap();
        out.write_f32::<E>(self.rotation_euler.z).unwrap();
        out.write_f32::<E>(self.translation.y).unwrap();
        out.write_f32::<E>(-self.translation.x).unwrap();
        out.write_f32::<E>(-self.translation.z).unwrap();
        for &c in self.landmark_confidence {
            out.write_f32::<E>(c).unwrap();
        }
        for &p in self.landmarks {
            out.write_f32::<E>(p.y).unwrap();
            out.write_f32::<E>(p.x).unwrap();
        }
        for p in self.landmarks_3d {
            out.write_f32::<E>(p.x).unwrap();
            out.write_f32::<E>(-p.y).unwrap();
            out.write_f32::<E>(-p.z).unwrap();
        }
        out.write_f32::<E>(self.features.eye_l).unwrap();
        out.write_f32::<E>(self.features.eye_r).unwrap();
        out.write_f32::<E>(self.features.eyebrow_updown_l).unwrap();
        out.write_f32::<E>(self.features.eyebrow_updown_r).unwrap();
        out.write_f32::<E>(self.features.eyebrow_quirk_l).unwrap();
        out.write_f32::<E>(self.features.eyebrow_quirk_r).unwrap();
        out.write_f32::<E>(self.features.eyebrow_steepness_l).unwrap();
        out.write_f32::<E>(self.features.eyebrow_steepness_r).unwrap();
        out.write_f32::<E>(self.features.mouth_corner_updown_l).unwrap();
        out.write_f32::<E>(self.features.mouth_corner_updown_r).unwrap();
        out.write_f32::<E>(self.features.mouth_corner_inout_l).unwrap();
        out.write_f32::<E>(self.features.mouth_corner_inout_r).unwrap();
        out.write_f32::<E>(self.features.mouth_open).unwrap();
        out.write_f32::<E>(self.features.mouth_wide).unwrap();
    }

    /// Create a [`FaceUpdate`] from a given [`TrackedFace`] and OpenSeeFace
    /// [`Features`].
    ///
    /// `width` & `height` are the image dimensions.
    /// `time` should be positive and in seconds.
    pub fn from_tracked_face<'a>(
        face: &'a TrackedFace,
        features: &'a Features,
        width: f32,
        height: f32,
        time: f64,
    ) -> FaceUpdate<'a> {
        let (rx, ry, rz) = face.rotation().to_euler(EulerRot::XYZ);
        let euler = vec3(rx, ry, rz);

        let blink_left = (1. + features.eye_l).clamp(0., 1.);
        let blink_right = (1. + features.eye_r).clamp(0., 1.);

        FaceUpdate {
            timestamp: time,
            face_id: face.id(),
            width,
            height,
            success: face.is_alive(),
            pnp_error: face.pose_error(),
            blink_left,
            blink_right,
            rotation: face.rotation(),
            rotation_euler: euler,
            translation: face.translation(),
            landmark_confidence: face.landmark_confidence(),
            landmarks: face.landmarks_image(),
            landmarks_3d: face.face_3d(),
            features,
        }
    }
}
