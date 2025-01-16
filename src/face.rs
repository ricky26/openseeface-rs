use glam::{vec3, Mat3, Vec2, Vec3};
use remedian::RemedianBlock;
use serde::{Deserialize, Serialize};

use crate::geometry::align_points;

pub const DEFAULT_FACE: [Vec3; 70] = [
    vec3(0.4551769692672, 0.300895790030204, -0.764429433974752),
    vec3(0.448998827123556, 0.166995837790733, -0.765143004071253),
    vec3(0.437431554952677, 0.022655479179981, -0.739267175112735),
    vec3(0.415033422928434, -0.088941454648772, -0.747947437846473),
    vec3(0.389123587370091, -0.232380029794684, -0.704788385327458),
    vec3(0.334630113904382, -0.361265387599081, -0.615587579236862),
    vec3(0.263725112132858, -0.460009725616771, -0.491479221041573),
    vec3(0.16241621322721, -0.558037146073869, -0.339445180872282),
    vec3(0., -0.621079019321682, -0.287294770748887),
    vec3(-0.16241621322721, -0.558037146073869, -0.339445180872282),
    vec3(-0.263725112132858, -0.460009725616771, -0.491479221041573),
    vec3(-0.334630113904382, -0.361265387599081, -0.615587579236862),
    vec3(-0.389123587370091, -0.232380029794684, -0.704788385327458),
    vec3(-0.415033422928434, -0.088941454648772, -0.747947437846473),
    vec3(-0.437431554952677, 0.022655479179981, -0.739267175112735),
    vec3(-0.448998827123556, 0.166995837790733, -0.765143004071253),
    vec3(-0.4551769692672, 0.300895790030204, -0.764429433974752),
    vec3(0.385529968662985, 0.402800553948697, -0.310031082540741),
    vec3(0.322196658344302, 0.464439136821772, -0.250558059367669),
    vec3(0.25409760441282, 0.46420381416882, -0.208177722146526),
    vec3(0.186875436782135, 0.44706071961879, -0.145299823706503),
    vec3(0.120880983543622, 0.423566314072968, -0.110757158774771),
    vec3(-0.120880983543622, 0.423566314072968, -0.110757158774771),
    vec3(-0.186875436782135, 0.44706071961879, -0.145299823706503),
    vec3(-0.25409760441282, 0.46420381416882, -0.208177722146526),
    vec3(-0.322196658344302, 0.464439136821772, -0.250558059367669),
    vec3(-0.385529968662985, 0.402800553948697, -0.310031082540741),
    vec3(0., 0.293332603215811, -0.137582088779393),
    vec3(0., 0.194828701837823, -0.069158109325951),
    vec3(0., 0.103844017393155, -0.009151819844964),
    vec3(0., 0., 0.),
    vec3(0.080626352317973, -0.041276068128093, -0.134161035564826),
    vec3(0.046439347377934, -0.057675223874769, -0.102990627164664),
    vec3(0., -0.068753126205604, -0.090545348482397),
    vec3(-0.046439347377934, -0.057675223874769, -0.102990627164664),
    vec3(-0.080626352317973, -0.041276068128093, -0.134161035564826),
    vec3(0.315905195966084, 0.298337502555443, -0.285107407636464),
    vec3(0.275252345439353, 0.312721904921771, -0.244558251170671),
    vec3(0.176394511553111, 0.311907184376107, -0.219205360345231),
    vec3(0.131229723798772, 0.284447361805627, -0.234239149487417),
    vec3(0.184124948330084, 0.260179585304867, -0.226590776513707),
    vec3(0.279433549294448, 0.267363071770222, -0.248441437111633),
    vec3(-0.131229723798772, 0.284447361805627, -0.234239149487417),
    vec3(-0.176394511553111, 0.311907184376107, -0.219205360345231),
    vec3(-0.275252345439353, 0.312721904921771, -0.244558251170671),
    vec3(-0.315905195966084, 0.298337502555443, -0.285107407636464),
    vec3(-0.279433549294448, 0.267363071770222, -0.248441437111633),
    vec3(-0.184124948330084, 0.260179585304867, -0.226590776513707),
    vec3(0.121155252430729, -0.208988660580347, -0.160606287940521),
    vec3(0.041356305910044, -0.194484199722098, -0.096159882202821),
    vec3(0., -0.205180167345702, -0.083299217789729),
    vec3(-0.041356305910044, -0.194484199722098, -0.096159882202821),
    vec3(-0.121155252430729, -0.208988660580347, -0.160606287940521),
    vec3(-0.132325402795928, -0.290857984604968, -0.187067868218105),
    vec3(-0.064137791831655, -0.325377847425684, -0.158924039726607),
    vec3(0., -0.343742581679188, -0.113925986025684),
    vec3(0.064137791831655, -0.325377847425684, -0.158924039726607),
    vec3(0.132325402795928, -0.290857984604968, -0.187067868218105),
    vec3(0.181481567104525, -0.243239316141725, -0.231284988892766),
    vec3(0.083999507750469, -0.239717753728704, -0.155256465640701),
    vec3(0., -0.256058040176369, -0.0950619498899),
    vec3(-0.083999507750469, -0.239717753728704, -0.155256465640701),
    vec3(-0.181481567104525, -0.243239316141725, -0.231284988892766),
    vec3(-0.074036069749345, -0.250689938345682, -0.177346470406188),
    vec3(0., -0.264945854681568, -0.112349967428413),
    vec3(0.074036069749345, -0.250689938345682, -0.177346470406188),
    // Pupils and eyeball centers
    vec3(0.257990002632141, 0.276080012321472, -0.219998998939991),
    vec3(-0.257990002632141, 0.276080012321472, -0.219998998939991),
    vec3(0.257990002632141, 0.276080012321472, -0.324570998549461),
    vec3(-0.257990002632141, 0.276080012321472, -0.324570998549461),
];

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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Features {
    pub eye_l: f32,
    pub eye_r: f32,
    pub eyebrow_updown_l: f32,
    pub eyebrow_updown_r: f32,
    pub eyebrow_quirk_l: f32,
    pub eyebrow_quirk_r: f32,
    pub eyebrow_steepness_l: f32,
    pub eyebrow_steepness_r: f32,
    pub mouth_corner_updown_l: f32,
    pub mouth_corner_updown_r: f32,
    pub mouth_corner_inout_l: f32,
    pub mouth_corner_inout_r: f32,
    pub mouth_open: f32,
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
        points: &[Vec2],
        offset: usize,
        norm_distance_y: f32,
        now: f64,
    ) -> f32 {
        let mut f_pts = [
            points[offset + 1],
            points[offset + 2],
            points[offset + 5],
            points[offset + 4],
        ];
        let a = align_points(points[offset], points[offset + 3], &mut f_pts);
        let f = (f_pts[0].y + f_pts[1].y - f_pts[2].y - f_pts[3].y) / (2. * norm_distance_y);
        *feature = tracker.update(f, now);
        a
    }

    fn update_eyebrow(
        steepness_tracker: &mut FeatureTracker,
        steepness: &mut f32,
        quirk_tracker: &mut FeatureTracker,
        quirk: &mut f32,
        points: &[Vec2],
        offset: usize,
        norm_angle: f32,
        norm_distance_y: f32,
        now: f64,
    ) {
        let mut f_pts = [
            points[offset],
            points[offset + 1],
            points[offset + 2],
            points[offset + 3],
            points[offset + 4],
        ];
        let a = align_points(points[offset], points[offset + 4], &mut f_pts);
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

    pub fn update(&mut self, points: &[Vec2], now: f64, full: bool) -> Features {
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
            let a3 = align_points(points[0], points[16], &mut []);
            let a4 = align_points(points[31], points[35], &mut []);
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

pub struct TrackedFace {
    pub(crate) id: usize,
    pub(crate) alive: bool,
    pub(crate) frame_count: usize,
    pub(crate) position: Vec2,
    pub(crate) landmarks: Vec<(Vec2, f32)>,
    pub(crate) face_3d: [Vec3; 70],
    pub(crate) contour_2d: Vec<Vec2>,
    pub(crate) translation: Vec3,
    pub(crate) rotation: Mat3,
}

impl TrackedFace {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }

    pub fn position(&self) -> Vec2 {
        self.position
    }

    pub fn landmarks(&self) -> &[(Vec2, f32)] {
        &self.landmarks
    }

    pub(crate) fn contour_2d(&self) -> &[Vec2] {
        &self.contour_2d
    }

    pub fn translation(&self) -> Vec3 {
        self.translation
    }

    pub fn rotation(&self) -> Mat3 {
        self.rotation
    }

    pub fn new(id: usize) -> TrackedFace {
        let face_3d = DEFAULT_FACE;

        TrackedFace {
            id,
            alive: false,
            position: Vec2::ZERO,
            frame_count: 0,
            landmarks: Vec::with_capacity(70),
            face_3d,
            contour_2d: Vec::new(),
            translation: Vec3::ZERO,
            rotation: Mat3::IDENTITY,
        }
    }

    pub(crate) fn update_contour(&mut self, indices: &[usize]) {
        self.contour_2d.clear();
        self.contour_2d.extend(indices.iter()
            .map(|&idx| self.landmarks[idx].0));
    }

    pub fn reset(&mut self) {
        self.alive = false;
    }
}
