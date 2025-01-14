use std::io::Write;
use byteorder::{ByteOrder, WriteBytesExt};
use glam::{Quat, Vec3};

pub const ALL_FEATURES: [&'static str; 14] = [
    "eye_l",
    "eye_r",
    "eyebrow_steepness_l",
    "eyebrow_updown_l",
    "eyebrow_quirk_l",
    "eyebrow_steepness_r",
    "eyebrow_updown_r",
    "eyebrow_quirk_r",
    "mouth_corner_updown_l",
    "mouth_corner_inout_l",
    "mouth_corner_updown_r",
    "mouth_corner_inout_r",
    "mouth_open",
    "mouth_wide",
];

pub struct FaceUpdate<'a> {
    pub timestamp: f64,
    pub face_id: u32,
    pub width: f32,
    pub height: f32,
    pub success: bool,
    pub pnp_error: f32,
    pub blink_left: f32,
    pub blink_right: f32,
    pub rotation: Quat,
    pub rotation_euler: Vec3,
    pub translation: Vec3,
    pub landmarks: &'a [Vec3],
    pub landmarks_3d: &'a [Vec3],
    pub features: &'a [f32],
}

impl FaceUpdate<'_> {
    pub fn write<E: ByteOrder>(&self, out: &mut impl Write) {
        out.write_f64::<E>(self.timestamp).unwrap();
        out.write_u32::<E>(self.face_id).unwrap();
        out.write_f32::<E>(self.width).unwrap();
        out.write_f32::<E>(self.height).unwrap();
        out.write_u8(if self.success { 1 } else { 0 }).unwrap();
        out.write_f32::<E>(self.pnp_error).unwrap();
        out.write_f32::<E>(self.rotation.x).unwrap();
        out.write_f32::<E>(self.rotation.y).unwrap();
        out.write_f32::<E>(self.rotation.z).unwrap();
        out.write_f32::<E>(self.rotation.w).unwrap();
        out.write_f32::<E>(self.rotation_euler.x).unwrap();
        out.write_f32::<E>(self.rotation_euler.y).unwrap();
        out.write_f32::<E>(self.rotation_euler.z).unwrap();
        out.write_f32::<E>(self.translation.x).unwrap();
        out.write_f32::<E>(self.translation.y).unwrap();
        out.write_f32::<E>(self.translation.z).unwrap();
        for p in self.landmarks {
            out.write_f32::<E>(p.z).unwrap();
        }
        for p in self.landmarks {
            out.write_f32::<E>(p.y).unwrap();
            out.write_f32::<E>(p.x).unwrap();
        }
        for p in self.landmarks_3d {
            out.write_f32::<E>(p.x).unwrap();
            out.write_f32::<E>(-p.y).unwrap();
            out.write_f32::<E>(-p.z).unwrap();
        }
        for &f in self.features {
            out.write_f32::<E>(f).unwrap();
        }
    }
}
