use glam::{uvec2, vec2, vec3, Mat3, Quat, UVec2, Vec2, Vec3};
use remedian::RemedianBlock;
use serde::{Deserialize, Serialize};
use sqpnp::{SqPnPSolution, SqPnPSolver};

fn align_points(a: Vec2, b: Vec2, points: &mut [Vec2]) -> f32 {
    let angle = (b - a).to_angle();
    let rot = Vec2::from_angle(-angle);
    for point in points {
        *point = a + rot.rotate(*point - a);
    }
    angle
}

pub const NUM_FACE_LANDMARKS: usize = 70;
pub type FaceLandmarks3d = [Vec3; NUM_FACE_LANDMARKS];
#[allow(clippy::excessive_precision)]
pub const DEFAULT_FACE: FaceLandmarks3d = [
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

pub const FACE_EDGES: [(usize, usize); 66] = [
    // Jaw
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    // Brow L
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),

    // Brow R
    (22, 23),
    (23, 24),
    (24, 25),
    (25, 26),

    // Nose
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 33),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),

    // Eye L
    (36, 37),
    (37, 38),
    (38, 39),
    (39, 40),
    (40, 41),
    (41, 36),

    // Eye R
    (42, 43),
    (43, 44),
    (44, 45),
    (45, 46),
    (46, 47),
    (47, 42),

    // Lips
    (48, 49),
    (49, 50),
    (50, 51),
    (51, 52),
    (52, 62),
    (62, 53),
    (53, 54),
    (54, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    (58, 48),

    (58, 59),
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
    (65, 58),

    // Gaze
    (66, 68),
    (67, 69),
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

pub struct TrackedFace {
    id: u32,
    alive: bool,
    has_pnp: bool,
    pnp_error: f32,
    pnp_solver: SqPnPSolver,
    frame_count: usize,
    centre: Vec2,
    size: Vec2,
    image_min: UVec2,
    image_max: UVec2,
    landmark_confidence: Vec<f32>,
    landmarks_image: Vec<Vec2>,
    landmarks_camera: Vec<Vec2>,
    face_3d: FaceLandmarks3d,
    contour_indices: &'static [usize],
    contour_3d: Vec<Vec3>,
    contour_2d: Vec<Vec2>,
    translation: Vec3,
    rotation_matrix: Mat3,
    rotation: Quat,
    inv_rotation: Quat,
    features: FeatureExtractor,
    current_features: Features,
}

impl TrackedFace {
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }

    pub fn centre(&self) -> Vec2 {
        self.centre
    }

    pub fn size(&self) -> Vec2 {
        self.size
    }

    pub fn image_min(&self) -> UVec2 {
        self.image_min
    }

    pub fn image_max(&self) -> UVec2 {
        self.image_max
    }

    pub fn landmark_confidence(&self) -> &[f32] {
        &self.landmark_confidence
    }

    pub fn landmarks_image(&self) -> &[Vec2] {
        &self.landmarks_image
    }

    pub fn landmarks_camera(&self) -> &[Vec2] {
        &self.landmarks_camera
    }

    pub fn has_pose(&self) -> bool {
        self.has_pnp
    }

    pub fn pose_error(&self) -> f32 {
        self.pnp_error
    }

    pub fn pose_solutions(&self) -> &[SqPnPSolution] {
        self.pnp_solver.solutions()
    }

    pub fn translation(&self) -> Vec3 {
        self.translation
    }

    pub fn rotation_matrix(&self) -> Mat3 {
        self.rotation_matrix
    }

    pub fn rotation(&self) -> Quat {
        self.rotation
    }

    pub fn face_3d(&self) -> &[Vec3] {
        &self.face_3d
    }

    pub fn features(&self) -> &Features {
        &self.current_features
    }

    pub(crate) fn new(
        id: u32,
        contour_indices: &'static [usize],
        max_feature_updates: f64,
    ) -> TrackedFace {
        let pnp_solver = SqPnPSolver::new();
        let face_3d = DEFAULT_FACE;
        let contour_3d = contour_indices.iter()
            .map(|&idx| face_3d[idx])
            .collect();

        let features = FeatureExtractor::new(max_feature_updates);

        TrackedFace {
            id,
            alive: false,
            has_pnp: false,
            pnp_error: f32::MAX,
            pnp_solver,
            centre: Vec2::ZERO,
            size: Vec2::ZERO,
            image_min: UVec2::ZERO,
            image_max: UVec2::ZERO,
            frame_count: 0,
            landmark_confidence: Vec::with_capacity(NUM_FACE_LANDMARKS),
            landmarks_image: Vec::with_capacity(NUM_FACE_LANDMARKS),
            landmarks_camera: Vec::with_capacity(NUM_FACE_LANDMARKS),
            face_3d,
            contour_indices,
            contour_3d,
            contour_2d: Vec::new(),
            translation: Vec3::ZERO,
            rotation_matrix: Mat3::IDENTITY,
            rotation: Quat::IDENTITY,
            inv_rotation: Quat::IDENTITY,
            features,
            current_features: Default::default(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.alive = false;
        self.has_pnp = false;
        self.pnp_error = f32::MAX;
        self.frame_count = 0;
    }

    fn update_contour(&mut self) {
        self.contour_2d.clear();
        self.contour_2d.extend(self.contour_indices.iter()
            .map(|&idx| self.landmarks_camera[idx]));
    }

    fn update_landmarks_camera(&mut self, width: u32, height: u32) {
        let size = width.min(height) as f32;
        let scale = 1. / size;
        let offset = uvec2(width, height).as_vec2() * 0.5 * scale;
        self.landmarks_camera.clear();
        self.landmarks_camera.extend(
            self.landmarks_image.iter()
                .map(|&p| (p * scale - offset) * vec2(0.5, -0.5)));
    }

    fn update_eye_3d(&mut self, offset: usize, outer_index: usize) {
        let outer_lm = self.landmarks_camera[outer_index];
        let outer_3d = self.face_3d[outer_index];
        let inner_index = outer_index + 3;
        let inner_lm = self.landmarks_camera[inner_index];
        let inner_3d = self.face_3d[inner_index];
        let pupil_lm = self.landmarks_camera[offset];

        // Update pupil 3D point.
        let d1 = (pupil_lm - outer_lm).length();
        let d2 = (pupil_lm - inner_lm).length();
        let pt = (outer_3d * d1 + inner_3d * d2) / (d1 + d2);
        let r = self.rotation * pt + self.translation;
        let pupil_3d = pupil_lm.extend(1.) * r.z;
        let pupil_3d = self.inv_rotation.mul_vec3(pupil_3d - self.translation);
        self.face_3d[offset] = pupil_3d;

        // Update eye centre.
        let eye_centre = (inner_3d + outer_3d) * 0.5;
        let d_corner = (inner_3d - outer_3d).length();
        let depth = 0.385 * d_corner;
        let eye_3d = eye_centre - Vec3::Z * depth;
        self.face_3d[offset + 2] = eye_3d;
    }

    fn update_face_3d(&mut self) {
        let r = &self.rotation;
        let ir = &self.inv_rotation;
        let t = self.translation;
        let z_plane = r.mul_vec3(Vec3::Z);
        let c0 = z_plane.dot(t);
        let mut err = 0.;

        for (index, ((p, &lm), &fp)) in self.face_3d[0..66].iter_mut()
            .zip(&self.landmarks_camera)
            .zip(&DEFAULT_FACE)
            .enumerate() {
            let c = fp.z + c0;
            let np = lm.extend(1.);
            let cp = z_plane.dot(np);
            let z = c / cp;
            let np = np * z;
            *p = ir.mul_vec3(np - t);

            if index < 17 || index == 30 {
                let rp = r.mul_vec3(fp) + t;
                let rp = rp.truncate() / rp.z;
                let np = np.truncate() / np.z;
                err += (np - rp).length_squared();
            }
        }

        self.update_eye_3d(66, 36);
        self.update_eye_3d(67, 42);

        let err = (err / (2. * (self.landmarks_image.len() as f32))).sqrt();
        self.pnp_error = err;
        if self.pnp_error > 300. {
            tracing::warn!("significant PnP error");
        }
    }

    pub(crate) fn update(
        &mut self,
        frame_width: u32,
        frame_height: u32,
        image_min: UVec2,
        image_max: UVec2,
        centre: Vec2,
        size: Vec2,
        confidence: &mut Vec<f32>,
        landmarks: &mut Vec<Vec2>,
        now: f64,
    ) {
        if self.alive {
            self.frame_count += 1;
        } else {
            self.frame_count = 1;
        }

        self.alive = true;
        self.centre = centre;
        self.size = size;
        self.image_min = image_min;
        self.image_max = image_max;
        self.landmark_confidence.clear();
        self.landmarks_image.clear();
        std::mem::swap(&mut self.landmark_confidence, confidence);
        std::mem::swap(&mut self.landmarks_image, landmarks);

        self.update_landmarks_camera(frame_width, frame_height);
        self.update_contour();

        let has_pnp = self.pnp_solver.solve(&self.contour_3d, &self.contour_2d, None);
        self.has_pnp = has_pnp;
        if !has_pnp {
            return;
        }

        let solution = self.pnp_solver.best_solution().unwrap();
        let t = solution.translation();
        let rot = solution.rotation_matrix();
        let q = Quat::from_mat3(&rot);
        self.has_pnp = true;
        self.translation = t;
        self.rotation_matrix = rot;
        self.rotation = q;
        self.inv_rotation = q.inverse();

        self.update_face_3d();

        self.current_features = self.features.update(&self.face_3d, now, true);
    }
}
