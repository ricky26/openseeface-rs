use glam::{Vec2, Vec3};
use remedian::RemedianBlock;
use serde::{Deserialize, Serialize};
use crate::face::{FaceLandmarks3d, TrackedFace};

fn align_points(a: Vec2, b: Vec2, points: &mut [Vec2]) -> f32 {
    let angle = (b - a).to_angle();
    let rot = Vec2::from_angle(-angle);
    for point in points {
        *point = a + rot.rotate(*point - a);
    }
    angle
}

#[derive(Clone, Debug)]
struct FeatureConfig {
    pub threshold: f32,
    pub alpha: f32,
    pub hard_factor: f32,
    pub decay: f32,
    pub max_feature_updates: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig {
            threshold: 0.15,
            alpha: 0.2,
            hard_factor: 0.15,
            decay: 0.001,
            max_feature_updates: 0.,
        }
    }
}

#[derive(Clone, Debug)]
struct FeatureTracker {
    median: RemedianBlock<f32>,
    min: Option<f32>,
    max: Option<f32>,
    hard_min: Option<f32>,
    hard_max: Option<f32>,
    last: f32,
    current_median: f32,
    first_seen: Option<f64>,
    updating: bool,
    config: FeatureConfig,
}

impl FeatureTracker {
    pub fn new(config: FeatureConfig) -> FeatureTracker {
        FeatureTracker {
            median: Default::default(),
            min: None,
            max: None,
            hard_min: None,
            hard_max: None,
            last: 0.0,
            current_median: 0.0,
            first_seen: None,
            updating: true,
            config,
        }
    }

    fn update_state(&mut self, x: f32, now: f64) -> f32 {
        self.updating = self.updating
            && self.first_seen.map_or(true, |first_seen| {
            now - first_seen < self.config.max_feature_updates
        });
        if self.updating {
            self.median.add_sample_point(x);
            self.current_median = self.median.median_or_default();
        }

        let median = self.current_median;
        if let Some(min) = self.min {
            if x < min {
                if self.updating {
                    self.min = Some(x);
                    self.hard_min = Some(x + self.config.hard_factor * (median - x));
                }
                return -1.;
            }
        } else {
            if x < median && (median - x) / median > self.config.threshold {
                if self.updating {
                    self.min = Some(x);
                    self.hard_min = Some(x + self.config.hard_factor * (median - x));
                }
                return -1.;
            }
            return 0.;
        }

        if let Some(max) = self.max {
            if x > max {
                if self.updating {
                    self.max = Some(x);
                    self.hard_max = Some(x - self.config.hard_factor * (x - median));
                }
                return 1.;
            }
        } else {
            if x > median && (x - median) / median > self.config.threshold {
                if self.updating {
                    self.max = Some(x);
                    self.hard_max = Some(x - self.config.hard_factor * (x - median));
                }
                return 1.;
            }
            return 0.;
        }

        if self.updating {
            let min = self.min.unwrap();
            let hard_min = self.hard_min.unwrap();
            let max = self.max.unwrap();
            let hard_max = self.hard_max.unwrap();

            if min < hard_min {
                self.min = Some(hard_min * self.config.decay + min * (1. - self.config.decay));
            }

            if max > hard_max {
                self.max = Some(hard_max * self.config.decay + max * (1. - self.config.decay));
            }
        }

        if x < median {
            let min = self.min.unwrap();
            -(1. - (x - min) / (median - min))
        } else if x > median {
            let max = self.max.unwrap();
            (x - median) / (max - median)
        } else {
            0.
        }
    }

    pub fn update(&mut self, x: f32, now: f64) -> f32 {
        if self.config.max_feature_updates > 0. && self.first_seen.is_none() {
            self.first_seen = Some(now);
        }

        let new = self.update_state(x, now);
        let filtered = self.last * self.config.alpha + new * (1. - self.config.alpha);
        self.last = filtered;
        filtered
    }
}

/// OpenSeeFace tracking features.
///
/// All features are in range 0..1.
///
/// Left/right here refer to the elements to the left/right of the image.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Features {
    /// Left eye open amount.
    pub eye_l: f32,
    /// Right eye open amount.
    pub eye_r: f32,
    /// Left eyebrow raise amount.
    pub eyebrow_updown_l: f32,
    /// Right eyebrow raise amount.
    pub eyebrow_updown_r: f32,
    /// Left eyebrow quirk amount (greater value means more deviation from flat eyebrow shape).
    pub eyebrow_quirk_l: f32,
    /// Right eyebrow quirk amount (greater value means more deviation from flat eyebrow shape).
    pub eyebrow_quirk_r: f32,
    /// Left eyebrow steepness (greater means steeper).
    pub eyebrow_steepness_l: f32,
    /// Right eyebrow steepness (greater means steeper).
    pub eyebrow_steepness_r: f32,
    /// Left mouth corner lower amount.
    pub mouth_corner_updown_l: f32,
    /// Right mouth corner lower amount.
    pub mouth_corner_updown_r: f32,
    /// Left mouth corner outwards amount.
    pub mouth_corner_inout_l: f32,
    /// Right mouth corner outwards amount.
    pub mouth_corner_inout_r: f32,
    /// Greater if mouth is more open.
    pub mouth_open: f32,
    /// Greater if mouth is horizontally stretched.
    pub mouth_wide: f32,
}

#[derive(Clone, Debug)]
struct FeatureExtractor {
    eye_l: FeatureTracker,
    eye_r: FeatureTracker,
    eyebrow_updown_l: FeatureTracker,
    eyebrow_updown_r: FeatureTracker,
    eyebrow_quirk_l: FeatureTracker,
    eyebrow_quirk_r: FeatureTracker,
    eyebrow_steepness_l: FeatureTracker,
    eyebrow_steepness_r: FeatureTracker,
    mouth_corner_updown_l: FeatureTracker,
    mouth_corner_updown_r: FeatureTracker,
    mouth_corner_inout_l: FeatureTracker,
    mouth_corner_inout_r: FeatureTracker,
    mouth_open: FeatureTracker,
    mouth_wide: FeatureTracker,
}

impl FeatureExtractor {
    pub fn new(max_feature_updates: f64) -> FeatureExtractor {
        FeatureExtractor {
            eye_l: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            eye_r: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            eyebrow_updown_l: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            eyebrow_updown_r: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            eyebrow_quirk_l: FeatureTracker::new(FeatureConfig {
                threshold: 0.05,
                max_feature_updates,
                ..Default::default()
            }),
            eyebrow_quirk_r: FeatureTracker::new(FeatureConfig {
                threshold: 0.05,
                max_feature_updates,
                ..Default::default()
            }),
            eyebrow_steepness_l: FeatureTracker::new(FeatureConfig {
                threshold: 0.05,
                max_feature_updates,
                ..Default::default()
            }),
            eyebrow_steepness_r: FeatureTracker::new(FeatureConfig {
                threshold: 0.05,
                max_feature_updates,
                ..Default::default()
            }),
            mouth_corner_updown_l: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            mouth_corner_updown_r: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            mouth_corner_inout_l: FeatureTracker::new(FeatureConfig {
                threshold: 0.02,
                max_feature_updates,
                ..Default::default()
            }),
            mouth_corner_inout_r: FeatureTracker::new(FeatureConfig {
                threshold: 0.02,
                max_feature_updates,
                ..Default::default()
            }),
            mouth_open: FeatureTracker::new(FeatureConfig {
                max_feature_updates,
                ..Default::default()
            }),
            mouth_wide: FeatureTracker::new(FeatureConfig {
                threshold: 0.02,
                max_feature_updates,
                ..Default::default()
            }),
        }
    }

    fn update_eye(
        tracker: &mut FeatureTracker,
        feature: &mut f32,
        points: &[Vec3],
        offset: usize,
        norm_distance_y: f32,
        now: f64,
    ) -> f32 {
        let mut f_pts = [
            points[offset + 1].truncate(),
            points[offset + 2].truncate(),
            points[offset + 5].truncate(),
            points[offset + 4].truncate(),
        ];
        let a = align_points(points[offset + 3].truncate(), points[offset].truncate(), &mut f_pts);
        let f = (f_pts[0].y + f_pts[1].y - f_pts[2].y - f_pts[3].y) / (2. * norm_distance_y);
        *feature = tracker.update(f, now);
        a
    }

    #[allow(clippy::too_many_arguments)]
    fn update_eyebrow(
        steepness_tracker: &mut FeatureTracker,
        steepness: &mut f32,
        quirk_tracker: &mut FeatureTracker,
        quirk: &mut f32,
        points: &[Vec3],
        offset: usize,
        norm_angle: f32,
        norm_distance_y: f32,
        now: f64,
    ) {
        let mut f_pts = [
            points[offset].truncate(),
            points[offset + 1].truncate(),
            points[offset + 2].truncate(),
            points[offset + 3].truncate(),
            points[offset + 4].truncate(),
        ];
        let a = align_points(points[offset].truncate(), points[offset + 4].truncate(), &mut f_pts);
        let f = -a.to_degrees() - norm_angle;
        *steepness = steepness_tracker.update(f, now);
        let f = f_pts[1..4]
            .iter()
            .map(|v| (v.y - f_pts[0].y).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            / norm_distance_y;
        *quirk = quirk_tracker.update(f, now);
    }

    pub fn update(&mut self, points: &FaceLandmarks3d, now: f64, full: bool) -> Features {
        let mut features = Features::default();

        let norm_distance_x = (points[0].x - points[16].x + points[1].x - points[15].x) / 2.;
        let norm_distance_y = (points[27].y - points[30].y) / 3.;

        let a1 = Self::update_eye(
            &mut self.eye_l,
            &mut features.eye_l,
            points,
            42,
            norm_distance_y,
            now,
        );
        let a2 = Self::update_eye(
            &mut self.eye_r,
            &mut features.eye_r,
            points,
            36,
            norm_distance_y,
            now,
        );

        if full {
            let a3 = align_points(points[0].truncate(), points[16].truncate(), &mut []);
            let a4 = align_points(points[31].truncate(), points[35].truncate(), &mut []);
            let norm_angle = ((a1 + a2 + a3 + a4) / 4.).to_degrees();

            Self::update_eyebrow(
                &mut self.eyebrow_steepness_l,
                &mut features.eyebrow_steepness_l,
                &mut self.eyebrow_quirk_l,
                &mut features.eyebrow_quirk_l,
                points,
                22,
                norm_angle,
                norm_distance_y,
                now,
            );

            Self::update_eyebrow(
                &mut self.eyebrow_steepness_r,
                &mut features.eyebrow_steepness_r,
                &mut self.eyebrow_quirk_r,
                &mut features.eyebrow_quirk_r,
                points,
                17,
                norm_angle,
                norm_distance_y,
                now,
            );
        }

        let f = (((points[22].y + points[26].y) / 2.) - points[27].y) / norm_distance_y;
        features.eyebrow_updown_l = self.eyebrow_updown_l.update(f, now);

        let f = (((points[17].y + points[21].y) / 2.) - points[27].y) / norm_distance_y;
        features.eyebrow_updown_r = self.eyebrow_updown_r.update(f, now);

        let upper_mouth_line = (points[49].y + points[50].y + points[51].y) / 3.;
        let center_line = (points[50].x
            + points[60].x
            + points[27].x
            + points[30].x
            + points[64].x
            + points[55].x)
            / 6.;

        let f = (upper_mouth_line - points[62].y) / norm_distance_y;
        features.mouth_corner_updown_l = self.mouth_corner_updown_l.update(f, now);
        if full {
            let f = (center_line - points[62].x).abs() / norm_distance_x;
            features.mouth_corner_inout_l = self.mouth_corner_inout_l.update(f, now);
        }

        let f = (upper_mouth_line - points[58].y) / norm_distance_y;
        features.mouth_corner_updown_r = self.mouth_corner_updown_r.update(f, now);
        if full {
            let f = (center_line - points[58].x).abs() / norm_distance_x;
            features.mouth_corner_inout_r = self.mouth_corner_inout_r.update(f, now);
        }

        let a = (points[59].y + points[60].y + points[61].y) / 3.;
        let b = (points[63].y + points[64].y + points[65].y) / 3.;
        let f = (a - b).abs() / norm_distance_y;
        features.mouth_open = self.mouth_open.update(f, now);

        let f = (points[58].x - points[62].x).abs() / norm_distance_x;
        features.mouth_wide = self.mouth_wide.update(f, now);

        features
    }
}

pub struct TrackerFeatures {
    feature_extractors: Vec<FeatureExtractor>,
    current_features: Vec<Features>,
}

impl TrackerFeatures {
    pub fn current_features(&self) -> &[Features] {
        &self.current_features
    }

    pub fn new(max_faces: usize, max_feature_updates: f64) -> TrackerFeatures {
        let feature_extractors = vec![FeatureExtractor::new(max_feature_updates); max_faces];
        let current_features = vec![Features::default(); max_faces];
        TrackerFeatures {
            feature_extractors,
            current_features,
        }
    }

    pub fn update(&mut self, faces: &[TrackedFace], now: f64) {
        for ((face, extractor), features) in faces.iter()
            .zip(&mut self.feature_extractors)
            .zip(&mut self.current_features) {
            if face.has_pose() {
                *features = extractor.update(face.face_3d(), now, true);
            }
        }
    }
}
