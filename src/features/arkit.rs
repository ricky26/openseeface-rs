//! ARKit blend shape tracking.
//!
//! This is an incomplete feature extractor based on the ARKit blend shapes but
//! the OpenSeeFace tracking model.

use std::f32::consts::PI;

use glam::{vec2, Vec2};
use serde::{Deserialize, Serialize};

use super::FeatureExtractor as FeatureExtractorTrait;
use crate::face::TrackedFace;

fn positive_negative(v: f32) -> (f32, f32) {
    (v.clamp(0., 1.), (-v).clamp(0., 1.))
}

/// Calibration for the look range of an eye.
#[derive(Debug, Clone)]
pub struct EyeCalibration {
    /// The sine of the central angles.
    pub min: Vec2,
    /// The reciprocal of the sine of the angle between [`Self::min`] and the maximum look angle.
    pub scale: Vec2,
}

impl Default for EyeCalibration {
    fn default() -> Self {
        EyeCalibration {
            min: Vec2::ZERO,
            scale: Vec2::ONE,
        }
    }
}

impl EyeCalibration {
    /// Create an eye calibration from the maximum look range of the eye, with
    /// a zero origin angle.
    pub fn from_angle(angles: Vec2) -> EyeCalibration {
        let range = angles.map(f32::sin);
        let scale = 1. / range;
        EyeCalibration { min: Vec2::ZERO, scale }
    }

    /// Compute the `(in, up, out down)` features for an eye with the given 2D look vector.
    ///
    /// This vector should be the x & y parts of a 3D unit vector.
    pub fn calibrate(&self, look: Vec2) -> (f32, f32, f32, f32) {
        let values = (look - self.min) * self.scale;
        let (look_in, look_out) = positive_negative(values.x);
        let (look_up, look_down) = positive_negative(values.y);
        (look_in, look_up, look_out, look_down)
    }

    /// Compute the `(in, up, out down)` features for an eye with the given 2D look vector.
    ///
    /// This vector should be the x & y parts of a 3D unit vector.
    ///
    /// The vector will be flipped in x before use.
    pub fn calibrate_flipped(&self, look: Vec2) -> (f32, f32, f32, f32) {
        self.calibrate(look * vec2(-1., 1.))
    }
}

/// Simple 1D calibration configuration.
#[derive(Clone, Debug)]
pub struct Calibration1D {
    pub offset: f32,
    pub scale: f32,
}

impl Default for Calibration1D {
    fn default() -> Self {
        Calibration1D {
            offset: 0.,
            scale: 1.,
        }
    }
}

impl Calibration1D {
    /// Create a new calibration value with a given offset & scale.
    pub fn new(offset: f32, scale: f32) -> Calibration1D {
        Calibration1D { offset, scale }
    }

    /// Create a new calibration value with a given minimum and maximum.
    pub fn from_range(min: f32, max: f32) -> Calibration1D {
        let offset = min;
        let scale = 1. / (max - min);
        Calibration1D { offset, scale }
    }

    /// Create a new calibration with an offset and identity scale.
    pub fn from_offset(offset: f32) -> Calibration1D {
        Calibration1D { offset, ..Default::default() }
    }

    /// Create a new calibration with scale and no offset.
    pub fn from_scale(scale: f32) -> Calibration1D {
        Calibration1D { scale, ..Default::default() }
    }

    /// Calculate the feature value given the observed value.
    pub fn calibrate(&self, value: f32) -> f32 {
        (value - self.offset) * self.scale
    }
}

/// Trait for controlling how features are calibrated.
pub trait Calibration {
    /// Configuration needed for this kind of calibration.
    type Config;

    /// Create an instance of this calibration given the required config.
    fn from_config(config: Self::Config) -> Self;

    /// Update the calibration.
    ///
    /// This should be called for every frame with the current time in seconds.
    fn update(&mut self, now: f64);

    /// Calibrate the inner brow up/down value.
    fn brow_inner_up(&mut self, value: f32) -> f32;

    /// Calibrate the left brow up/down value.
    fn brow_outer_up_left(&mut self, value: f32) -> f32;

    /// Calibrate the right brow up/down value.
    fn brow_outer_up_right(&mut self, value: f32) -> f32;

    /// Calibrate the jaw left/right value.
    fn jaw_left_right(&mut self, value: f32) -> f32;

    /// Calibrate the jaw open value.
    fn jaw_open(&mut self, value: f32) -> f32;

    /// Calibrate the left eye blink value.
    fn eye_blink_left(&mut self, value: f32) -> f32;

    /// Calibrate the right eye blink value.
    fn eye_blink_right(&mut self, value: f32) -> f32;

    /// Calibrate the left eye look vector.
    fn eye_look_left(&mut self, value: Vec2) -> (f32, f32, f32, f32);

    /// Calibrate the right eye look vector.
    fn eye_look_right(&mut self, value: Vec2) -> (f32, f32, f32, f32);
}

/// Calibration based on static values which are passed in at construction time.
#[derive(Clone, Debug)]
pub struct StaticCalibration {
    pub brow_inner_up: Calibration1D,
    pub brow_outer_up_left: Calibration1D,
    pub brow_outer_up_right: Calibration1D,
    pub jaw_left_right: Calibration1D,
    pub jaw_open: Calibration1D,
    pub eye_blink_left: Calibration1D,
    pub eye_blink_right: Calibration1D,
    pub eye_look_left: EyeCalibration,
    pub eye_look_right: EyeCalibration,
}

impl Default for StaticCalibration {
    fn default() -> Self {
        StaticCalibration {
            brow_inner_up: Calibration1D::new(0.11, 1. / 0.04),
            brow_outer_up_left: Calibration1D::new(0.125, 1. / 0.05),
            brow_outer_up_right: Calibration1D::new(0.125, 1. / 0.05),
            jaw_left_right: Calibration1D::from_scale(1. / 0.004),
            jaw_open: Calibration1D::from_range(0.525, 0.625),
            eye_blink_left: Calibration1D::from_range(0.035, 0.025),
            eye_blink_right: Calibration1D::from_range(0.035, 0.025),
            eye_look_left: EyeCalibration::from_angle(vec2(0.15 * PI, 0.075 * PI)),
            eye_look_right: EyeCalibration::from_angle(vec2(0.15 * PI, 0.075 * PI)),
        }
    }
}

impl Calibration for StaticCalibration {
    type Config = Self;

    fn from_config(config: Self::Config) -> Self {
        config
    }

    fn update(&mut self, _now: f64) {}

    fn brow_inner_up(&mut self, value: f32) -> f32 {
        self.brow_inner_up.calibrate(value)
    }

    fn brow_outer_up_left(&mut self, value: f32) -> f32 {
        self.brow_outer_up_left.calibrate(value)
    }

    fn brow_outer_up_right(&mut self, value: f32) -> f32 {
        self.brow_outer_up_right.calibrate(value)
    }

    fn jaw_left_right(&mut self, value: f32) -> f32 {
        self.jaw_left_right.calibrate(value)
    }

    fn jaw_open(&mut self, value: f32) -> f32 {
        self.jaw_open.calibrate(value)
    }

    fn eye_blink_left(&mut self, value: f32) -> f32 {
        self.eye_blink_left.calibrate(value)
    }

    fn eye_blink_right(&mut self, value: f32) -> f32 {
        self.eye_blink_left.calibrate(value)
    }

    fn eye_look_left(&mut self, value: Vec2) -> (f32, f32, f32, f32) {
        self.eye_look_left.calibrate_flipped(value)
    }

    fn eye_look_right(&mut self, value: Vec2) -> (f32, f32, f32, f32) {
        self.eye_look_right.calibrate(value)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Features {
    pub brow_down_left: f32,
    pub brow_down_right: f32,
    pub brow_inner_up: f32,
    pub brow_outer_up_left: f32,
    pub brow_outer_up_right: f32,
    pub cheek_puff: f32,
    pub cheek_squint_left: f32,
    pub cheek_squint_right: f32,
    pub eye_blink_left: f32,
    pub eye_blink_right: f32,
    pub eye_look_down_left: f32,
    pub eye_look_down_right: f32,
    pub eye_look_in_left: f32,
    pub eye_look_in_right: f32,
    pub eye_look_out_left: f32,
    pub eye_look_out_right: f32,
    pub eye_look_up_left: f32,
    pub eye_look_up_right: f32,
    pub eye_squint_left: f32,
    pub eye_squint_right: f32,
    pub eye_wide_left: f32,
    pub eye_wide_right: f32,
    pub jaw_forward: f32,
    pub jaw_left: f32,
    pub jaw_open: f32,
    pub jaw_right: f32,
    pub mouth_close: f32,
    pub mouth_dimple_left: f32,
    pub mouth_dimple_right: f32,
    pub mouth_frown_left: f32,
    pub mouth_frown_right: f32,
    pub mouth_funnel: f32,
    pub mouth_left: f32,
    pub mouth_lower_down_left: f32,
    pub mouth_lower_down_right: f32,
    pub mouth_press_left: f32,
    pub mouth_press_right: f32,
    pub mouth_pucker: f32,
    pub mouth_right: f32,
    pub mouth_roll_lower: f32,
    pub mouth_roll_upper: f32,
    pub mouth_shrug_lower: f32,
    pub mouth_shrug_upper: f32,
    pub mouth_smile_left: f32,
    pub mouth_smile_right: f32,
    pub mouth_stretch_left: f32,
    pub mouth_stretch_right: f32,
    pub mouth_upper_up_left: f32,
    pub mouth_upper_up_right: f32,
    pub nose_sneer_left: f32,
    pub nose_sneer_right: f32,
}

/// Configuration for the ARKit feature extractor.
#[derive(Debug, Clone, Default)]
pub struct Config<C> {
    pub calibration: C,
}

/// An ARKit feature extractor.
///
/// This extracts ARKit blend shape coefficients.
///
/// This extractor does not produce all coefficients because some are not easily
/// observable from the OpenSeeFace model.
#[derive(Debug, Clone, Default)]
pub struct FeatureExtractor<C> {
    calibration: C,
}

impl<C: Calibration> FeatureExtractor<C> {
    pub fn update_raw(&mut self, features: &mut Features, face: &TrackedFace, now: f64) {
        self.calibration.update(now);

        let p = face.face_3d();

        features.brow_inner_up = (p[21].y + p[22].y) * 0.5 - p[27].y;

        features.brow_outer_up_left = (p[17].y + p[18].y + p[19].y) / 3. - p[27].y;
        features.brow_outer_up_right = (p[24].y + p[25].y + p[26].y) / 3. - p[27].y;

        let nose_mid_x = (p[27].x + p[28].x + p[29].x + p[30].x + p[33].x) / 5.;
        let jaw_mid_x = (p[0].x + p[8].x + p[16].x) / 3.;
        features.jaw_left = nose_mid_x - jaw_mid_x;
        features.jaw_open = (p[33] - p[8]).length();

        features.eye_blink_left = (p[37].y - p[41].y + p[38].y - p[40].y) * 0.5;
        features.eye_blink_right = (p[44].y - p[46].y + p[43].y - p[47].y) * 0.5;

        let eye_look_left = (p[66] - p[68]).normalize_or_zero();
        features.eye_look_out_left = eye_look_left.x;
        features.eye_look_up_left = eye_look_left.y;

        let eye_look_right = (p[67] - p[69]).normalize_or_zero();
        features.eye_look_out_right = eye_look_right.x;
        features.eye_look_up_right = eye_look_right.y;
    }

    pub fn calibrate(&mut self, features: &mut Features) {
        features.brow_inner_up = self.calibration.brow_inner_up(features.brow_inner_up)
            .clamp(0., 1.);

        let brow_outer_up_left = self.calibration.brow_outer_up_left(features.brow_outer_up_left);
        (features.brow_outer_up_left, features.brow_down_left) = positive_negative(brow_outer_up_left);

        let brow_outer_up_right = self.calibration.brow_outer_up_right(features.brow_outer_up_right);
        (features.brow_outer_up_right, features.brow_down_right) = positive_negative(brow_outer_up_right);

        let jaw_left_right = self.calibration.jaw_left_right(features.jaw_left);
        (features.jaw_left, features.jaw_right) = positive_negative(jaw_left_right);
        features.jaw_open = self.calibration.jaw_open(features.jaw_open).clamp(0., 1.);

        features.eye_blink_left = self.calibration.eye_blink_left(features.eye_blink_left)
            .clamp(0., 1.);
        features.eye_blink_right = self.calibration.eye_blink_right(features.eye_blink_right)
            .clamp(0., 1.);

        (
            features.eye_look_in_left,
            features.eye_look_up_left,
            features.eye_look_out_left,
            features.eye_look_down_left,
        ) = self.calibration.eye_look_left(vec2(
            features.eye_look_out_left,
            features.eye_look_up_left,
        ));

        (
            features.eye_look_in_right,
            features.eye_look_up_right,
            features.eye_look_out_right,
            features.eye_look_down_right,
        ) = self.calibration.eye_look_right(vec2(
            features.eye_look_out_right,
            features.eye_look_up_right,
        ));
    }
}

impl<C: Calibration> FeatureExtractorTrait for FeatureExtractor<C> {
    type Features = Features;
    type Config = Config<C::Config>;

    fn from_config(config: Self::Config) -> Self {
        FeatureExtractor {
            calibration: C::from_config(config.calibration),
        }
    }

    fn update(&mut self, features: &mut Self::Features, face: &TrackedFace, now: f64) {
        self.update_raw(features, face, now);
        self.calibrate(features);
    }
}
