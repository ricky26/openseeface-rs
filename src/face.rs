use glam::{uvec2, vec2, vec3, Mat3, Quat, UVec2, Vec2, Vec3};
use sqpnp::{SqPnPSolution, SqPnPSolver};

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

    pub fn face_3d(&self) -> &FaceLandmarks3d {
        &self.face_3d
    }

    pub(crate) fn new(
        id: u32,
        contour_indices: &'static [usize],
    ) -> TrackedFace {
        let pnp_solver = SqPnPSolver::new();
        let face_3d = DEFAULT_FACE;
        let contour_3d = contour_indices.iter()
            .map(|&idx| face_3d[idx])
            .collect();

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
    }
}
