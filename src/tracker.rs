use std::sync::LazyLock;

use glam::{uvec2, vec2, vec3, Mat2, UVec2, Vec2, Vec3};
use image::{GenericImage, GenericImageView, Rgb, Rgb32FImage, SubImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use ndarray::{s, ArrayViewD, Axis};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::face::TrackedFace;
use crate::image::ImageArrayExt;
use crate::retinaface::RetinaFaceDetector;

const MODEL_0: &[u8] = include_bytes!("../models/lm_model0_opt.onnx");
const MODEL_1: &[u8] = include_bytes!("../models/lm_model1_opt.onnx");
const MODEL_2: &[u8] = include_bytes!("../models/lm_model2_opt.onnx");
const MODEL_3: &[u8] = include_bytes!("../models/lm_model3_opt.onnx");
const MODEL_4: &[u8] = include_bytes!("../models/lm_model4_opt.onnx");
const MODEL_T: &[u8] = include_bytes!("../models/lm_modelT_opt.onnx");
const MODEL_V: &[u8] = include_bytes!("../models/lm_modelV_opt.onnx");
const MODEL_U: &[u8] = include_bytes!("../models/lm_modelU_opt.onnx");

const MODEL_GAZE: &[u8] = include_bytes!("../models/mnv3_gaze32_split_opt.onnx");

const MODEL_DETECTION: &[u8] = include_bytes!("../models/mnv3_detection_opt.onnx");

#[derive(Clone, Copy, Debug, Default, PartialOrd, Ord, PartialEq, Eq)]
pub enum TrackerModel {
    Model0,
    Model1,
    Model2,
    #[default]
    Model3,
    Model4,
    ModelT,
    ModelV,
    ModelU,
}

impl TrackerModel {
    pub fn from_i32(x: i32) -> Option<TrackerModel> {
        match x {
            0 => Some(TrackerModel::Model0),
            1 => Some(TrackerModel::Model1),
            2 => Some(TrackerModel::Model2),
            3 => Some(TrackerModel::Model3),
            4 => Some(TrackerModel::Model4),
            -1 => Some(TrackerModel::ModelT),
            -2 => Some(TrackerModel::ModelU),
            -3 => Some(TrackerModel::ModelV),
            _ => None,
        }
    }
}

fn model_data(model: TrackerModel) -> &'static [u8] {
    match model {
        TrackerModel::Model0 => MODEL_0,
        TrackerModel::Model1 => MODEL_1,
        TrackerModel::Model2 => MODEL_2,
        TrackerModel::Model3 => MODEL_3,
        TrackerModel::Model4 => MODEL_4,
        TrackerModel::ModelT => MODEL_T,
        TrackerModel::ModelV => MODEL_V,
        TrackerModel::ModelU => MODEL_U,
    }
}

pub const CONTOUR_INDICES: [usize; 14] = [0,1,8,15,16,27,28,29,30,31,32,33,34,35];
pub const CONTOUR_INDICES_T: [usize; 8] = [0,2,8,14,16,27,30,33];

fn logit(x: f32, factor: f32) -> f32 {
    let x = x.clamp(1e-7, 1. - 1e-7);
    let x = x / (1. - x);
    x.ln() / factor
}

struct ImagePreparer {
    std: Vec3,
    mean: Vec3,
    default: Vec3,
}

impl ImagePreparer {
    pub fn prepare_image_warp(
        &self, out: &mut Rgb32FImage, frame: &Rgb32FImage, projection: &Projection,
    ) {
        warp_into(
            frame,
            projection,
            Interpolation::Bilinear,
            Rgb(self.default.to_array()),
            out,
        );
        for pixel in out.pixels_mut() {
            pixel.0 = (Vec3::from_array(pixel.0) * self.std - self.mean).to_array();
        }
    }

    pub fn prepare_sub_image(&self, out: &mut Rgb32FImage, frame: &SubImage<&Rgb32FImage>) {
        let (x, y) = frame.offsets();
        let (w, h) = frame.dimensions();
        let projection = Projection::scale(
            (out.width() as f32) / (w as f32),
            (out.height() as f32) / (h as f32),
        ) * Projection::translate(-(x as f32), -(y as f32));
        self.prepare_image_warp(out, frame.inner(), &projection)
    }

    pub fn prepare_image(&self, out: &mut Rgb32FImage, frame: &Rgb32FImage) {
        let frame_size = uvec2(frame.width(), frame.height());
        let out_size = uvec2(out.width(), out.height());
        let base_scale = out_size.as_vec2() / frame_size.as_vec2();
        let scale = base_scale.x.min(base_scale.y);
        let offset = ((out_size.as_vec2() - frame_size.as_vec2() * scale) / 2.).floor();
        let projection = Projection::translate(offset.x, offset.y)
            * Projection::scale(scale, scale);
        self.prepare_image_warp(out, frame, &projection)
    }

    pub fn denormalise_image(&self, frame: &Rgb32FImage) -> Rgb32FImage {
        let mut result = frame.clone();
        for pixel in result.pixels_mut() {
            pixel.0 = ((Vec3::from_array(pixel.0) + self.mean) / self.std).to_array();
        }
        result
    }
}

static IMAGE_PREPARER: LazyLock<ImagePreparer> = LazyLock::new(|| {
    let mean = vec3(0.485, 0.456, 0.406);
    let std = vec3(0.229, 0.224, 0.225);
    let default = mean;
    let mean = mean / std;
    let std = 1. / std;
    ImagePreparer { std, mean, default }
});



#[derive(Clone, Debug)]
pub struct TrackerConfig {
    pub size: UVec2,
    pub model_type: TrackerModel,
    pub max_faces: usize,
    pub detection_threshold: f32,
    pub threshold: f32,
    pub discard_after: usize,
    pub scan_every: usize,
    pub bbox_growth: f32,
    pub max_threads: usize,
    pub no_gaze: bool,
    pub use_retinaface: bool,
    pub use_internal_face_detection: bool,
    pub assume_fullscreen_face: bool,
    pub static_model: bool,
    pub feature_level: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig {
            size: uvec2(640, 480),
            model_type: TrackerModel::Model0,
            max_faces: 1,
            detection_threshold: 0.6,
            threshold: 0.6,
            discard_after: 5,
            scan_every: 3,
            bbox_growth: 0.0,
            max_threads: 4,
            no_gaze: false,
            use_retinaface: false,
            use_internal_face_detection: true,
            assume_fullscreen_face: false,
            static_model: false,
            feature_level: 2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PendingFace {
    pub landmark_confidence: Vec<f32>,
    pub landmarks: Vec<Vec2>,
    pub disabled: bool,
    pub confidence: f32,
    pub centre: Vec2,
    pub size: Vec2,
    pub image_min: UVec2,
    pub image_max: UVec2,
    pub bounds_min: Vec2,
    pub bounds_max: Vec2,
}

impl PendingFace {
    fn update_bounds(&mut self) {
        let (min, max) = self.landmarks.iter()
            .fold(None, |acc, &p| {
                if let Some((min, max)) = acc {
                    Some((p.min(min), p.max(max)))
                } else {
                    Some((p, p))
                }
            })
            .unwrap();
        self.bounds_min = min;
        self.bounds_max = max;
        self.centre = (min + max) * 0.5;
        self.size = max - min;
    }
}

pub struct Tracker {
    config: TrackerConfig,
    res: u32,
    out_res: u32,
    logit_factor: f32,
    retinaface_detect: RetinaFaceDetector,
    retinaface_scan: RetinaFaceDetector,
    landmark_model: Session,
    gaze_model: Session,
    detect_model: Session,
    face_detect_224: Rgb32FImage,
    face_scratch_res: Rgb32FImage,
    eye_scratch_32: Rgb32FImage,
    eyes_scratch_32: Rgb32FImage,
    image_prep: &'static ImagePreparer,
    frame_count: usize,
    faces: Vec<TrackedFace>,
    tracked_faces: Vec<usize>,
    wait_count: usize,
    face_detections: Vec<(Vec2, Vec2)>,
    pending_faces: Vec<PendingFace>,
    num_pending_faces: usize,
    pending_face_indices: Vec<usize>,
}

impl Tracker {
    pub fn face_detections(&self) -> &[(Vec2, Vec2)] {
        &self.face_detections
    }

    pub fn faces(&self) -> &[TrackedFace] {
        &self.faces
    }

    pub fn num_visible_faces(&self) -> usize {
        self.tracked_faces.len()
    }

    pub fn visible_faces(&self) -> impl Iterator<Item = &TrackedFace> {
        self.tracked_faces.iter().map(|&idx| &self.faces[idx])
    }

    pub fn iter_debug_images(&self) -> impl Iterator<Item = (&'static str, &Rgb32FImage)> {
        [
            ("face_detect_224", &self.face_detect_224),
            ("face_scratch_res", &self.face_scratch_res),
            ("eyes_scratch_32", &self.eyes_scratch_32),
        ].into_iter()
    }

    pub fn denormalise_image(&self, frame: &Rgb32FImage) -> Rgb32FImage {
        self.image_prep.denormalise_image(frame)
    }

    pub fn new(
        config: TrackerConfig,
    ) -> Result<Tracker, ort::Error> {
        let retinaface_detect = RetinaFaceDetector::new(
            config.max_threads.max(4),
            0.4,
            0.4,
            config.max_faces,
        )?;
        let retinaface_scan = RetinaFaceDetector::new(
            2,
            0.4,
            0.4,
            config.max_faces,
        )?;

        let model_data = model_data(config.model_type);
        let landmark_model = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(config.max_threads.min(4))?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(model_data)?;
        let gaze_model = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(1)?
            .commit_from_memory(MODEL_GAZE)?;
        let detect_model = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(1)?
            .commit_from_memory(MODEL_DETECTION)?;

        let mut faces = Vec::with_capacity(config.max_faces);
        let tracked_faces = Vec::with_capacity(faces.len());
        let contour_indices = if config.model_type == TrackerModel::ModelT { &CONTOUR_INDICES_T[..] } else { &CONTOUR_INDICES };

        for i in 0..(config.max_faces as u32) {
            faces.push(TrackedFace::new(i, contour_indices));
        }

        let (res, out_res, logit_factor) = match config.model_type {
            TrackerModel::ModelT => (56, 7, 8.),
            TrackerModel::ModelU | TrackerModel::ModelV => (112, 14, 16.),
            _ => (224, 28, 16.),
        };

        let face_detect_224 = Rgb32FImage::new(224, 224);
        let face_scratch_res = Rgb32FImage::new(res, res);
        let eye_scratch_32 = Rgb32FImage::new(32, 32);
        let eyes_scratch_32 = Rgb32FImage::new(32, 64);
        let image_prep = &*IMAGE_PREPARER;

        let pending_faces = vec![PendingFace {
            landmarks: Vec::with_capacity(70),
            ..Default::default()
        }; config.max_faces];
        let pending_face_indices = Vec::with_capacity(pending_faces.len());

        Ok(Self {
            config,
            res,
            out_res,
            logit_factor,
            retinaface_detect,
            retinaface_scan,
            landmark_model,
            gaze_model,
            detect_model,
            face_detect_224,
            face_scratch_res,
            eye_scratch_32,
            eyes_scratch_32,
            image_prep,
            frame_count: 0,
            faces,
            tracked_faces,
            wait_count: 0,
            face_detections: Vec::new(),
            pending_faces,
            num_pending_faces: 0,
            pending_face_indices,
        })
    }

    #[tracing::instrument(skip_all)]
    fn detect_faces(&mut self, frame: &Rgb32FImage) -> Result<(), ort::Error> {
        self.image_prep.prepare_image(&mut self.face_detect_224, frame);
        let input = self.face_detect_224.as_view()
            .insert_axis(Axis(0))
            .permuted_axes((0, 3, 1, 2));
        let output = self.detect_model.run(ort::inputs![input]?)?;
        let max_pool = output["maxpool"].try_extract_tensor::<f32>()?;
        let output = output["output"].try_extract_tensor::<f32>()?;

        let frame_size = uvec2(frame.width(), frame.height());
        let square_size = frame_size.x.max(frame_size.y);
        let scale = (square_size as f32) / 224.;
        let offset = ((UVec2::splat(square_size) - frame_size) / 2).as_vec2();

        for x in 0..56 {
            for y in 0..56 {
                let c = output[[0, 0, y, x]];
                if c < self.config.detection_threshold {
                    continue;
                }

                let xc = max_pool[[0, 0, y, x]];
                if xc != c {
                    continue;
                }

                let r = output[[0, 1, y, x]] * 112.;
                let x = (x as f32 * 4. - r) * scale - offset.x;
                let y = (y as f32 * 4. - r) * scale - offset.y;
                let w = (2. * r) * scale;
                let h = (2. * r) * scale;
                let min = vec2(x, y);
                let max = vec2(w, h);
                self.face_detections.push((min, max));
            }
        }

        Ok(())
    }

    fn prepare_eye(
        &mut self,
        frame: &Rgb32FImage,
        inner: Vec2,
        outer: Vec2,
        flip: bool,
    ) -> (Vec2, Mat2) {
        let out_size = 32.;
        let half_out = out_size * 0.5;

        let angle = (inner - outer).to_angle();
        let center = (outer + inner) / 2.;
        let eye_size = (outer - inner).length();
        let src_size = vec2(1.6, 1.2) * eye_size;
        let base_scale = out_size / src_size;
        let tl = center - Vec2::from_angle(-angle).rotate(src_size * 0.5);
        let m = Mat2::from_scale_angle(1. / base_scale, -angle);
        let mut scale = base_scale;
        if flip {
            scale.x *= -1.;
        }
        let projection = Projection::translate(half_out, half_out)
            * Projection::scale(scale.x, scale.y)
            * Projection::rotate(-angle)
            * Projection::translate(-center.x, -center.y);
        self.image_prep.prepare_image_warp(&mut self.eye_scratch_32, frame, &projection);
        (tl, m)
    }

    fn process_eye(
        &self,
        results: &ArrayViewD<f32>,
        index: usize,
        offset: Vec2,
        m: Mat2,
    ) -> (Vec2, f32) {
        let t_c = results.slice(s![index, 0, .., ..]);
        let (x, y, c) = t_c.indexed_iter()
            .fold(None, |acc, ((y, x), &c)| {
                if let Some((prev_x, prev_y, prev_c)) = acc {
                    if c > prev_c {
                        Some((x, y, c))
                    } else {
                        Some((prev_x, prev_y, prev_c))
                    }
                } else {
                    Some((x, y, c))
                }
            })
            .unwrap();

        let x_off = results[[index, 2, y, x]];
        let y_off = results[[index, 1, y, x]];
        let x_off = 32. * logit(x_off, 8.);
        let y_off = 32. * logit(y_off, 8.);
        let x_eye = 4. * (x as f32) + x_off;
        let y_eye = 4. * (y as f32) + y_off;

        // Flip left eye
        let x_eye = if index > 0 { 32. - x_eye } else { x_eye };

        let rel = vec2(x_eye, y_eye);
        let p = offset + m.mul_vec2(rel);
        (p, c)
    }

    #[allow(clippy::type_complexity)]
    fn detect_eyes(
        &mut self, frame: &Rgb32FImage, face_index: usize,
    ) -> Result<((Vec2, f32), (Vec2, f32)), ort::Error> {
        if self.config.no_gaze {
            return Ok((Default::default(), Default::default()));
        }

        let landmarks = &self.pending_faces[face_index].landmarks;
        let inner_r = landmarks[39];
        let outer_r = landmarks[36];
        let inner_l = landmarks[45];
        let outer_l = landmarks[42];

        let (pos_r, m_r) = self.prepare_eye(frame, inner_r, outer_r, false);
        self.eyes_scratch_32.copy_from(&self.eye_scratch_32, 0, 0).unwrap();
        let (pos_l, m_l) = self.prepare_eye(frame, inner_l, outer_l, true);
        self.eyes_scratch_32.copy_from(&self.eye_scratch_32, 0, 32).unwrap();

        let input = self.eyes_scratch_32.as_view()
            .insert_axis(Axis(0))
            .into_shape_with_order((2, 32, 32, 3))
            .unwrap()
            .permuted_axes((0, 3, 1, 2));
        let outputs = self.gaze_model.run(ort::inputs![input]?)?;
        let results = outputs[0].try_extract_tensor::<f32>()?;

        let eye_r = self.process_eye(&results, 0, pos_r, m_r);
        let eye_l = self.process_eye(&results, 1, pos_l, m_l);
        Ok((eye_r, eye_l))
    }

    fn detect_landmarks(
        &mut self, frame: &Rgb32FImage, face_index: usize, min: Vec2, max: Vec2,
    ) -> Result<f32, ort::Error> {
        let size = uvec2(frame.width(), frame.height()).as_vec2();
        let min = min.clamp(Vec2::ZERO, size).as_uvec2();
        let max = (max + 1.).clamp(Vec2::ZERO, size).as_uvec2();
        let crop_size = max - min;
        let crop = frame.view(min.x, min.y, crop_size.x, crop_size.y);
        let scale = crop_size.as_vec2() / (self.res as f32);

        self.image_prep.prepare_sub_image(&mut self.face_scratch_res, &crop);
        let input = self.face_scratch_res.as_view()
            .insert_axis(Axis(0))
            .permuted_axes((0, 3, 1, 2));
        let outputs = self.landmark_model.run(ort::inputs![input]?)?;
        let result = outputs[0]
            .try_extract_tensor::<f32>()?;

        let (c0, c1, c2) = match self.config.model_type {
            TrackerModel::ModelT => (30usize, 60, 90),
            _ => (66, 132, 198),
        };

        let res_1 = (self.res - 1) as f32;
        let res_scale = res_1 / ((self.out_res - 1) as f32);
        let t_c = result.slice(s![0, 0..c0, .., ..]);
        let t_y = result.slice(s![0, c0..c1, .., ..]);
        let t_x = result.slice(s![0, c1..c2, .., ..]);

        let mut total_c = 0.;
        let face = &mut self.pending_faces[face_index];
        face.landmarks.clear();
        face.image_min = min;
        face.image_max = max;

        let min = min.as_vec2();
        for index in 0..c0 {
            let t_c = t_c.slice(s![index, .., ..]);
            let (x, y, c) = t_c.indexed_iter()
                .fold(None, |acc, ((y, x), &c)| {
                    if let Some((prev_x, prev_y, prev_c)) = acc {
                        if c > prev_c {
                            Some((x, y, c))
                        } else {
                            Some((prev_x, prev_y, prev_c))
                        }
                    } else {
                        Some((x, y, c))
                    }
                })
                .unwrap();

            let x_off = t_x[[index, y, x]];
            let y_off = t_y[[index, y, x]];
            let x_off = res_1 * logit(x_off, self.logit_factor);
            let y_off = res_1 * logit(y_off, self.logit_factor);
            let x_off = min.x + scale.x * (res_scale * (x as f32) + x_off + 0.5);
            let y_off = min.y + scale.y * (res_scale * (y as f32) + y_off + 0.5);
            let off = vec2(x_off, y_off);

            if self.config.model_type == TrackerModel::ModelT {
                unimplemented!();
            }

            total_c += c;
            face.landmark_confidence.push(c);
            face.landmarks.push(off);
        }

        let avg_c = total_c / (face.landmarks.len() as f32);
        Ok(avg_c)
    }

    #[tracing::instrument(skip_all)]
    pub fn detect(&mut self, frame: &Rgb32FImage) -> Result<(), ort::Error> {
        self.frame_count += 1;
        self.wait_count += 1;

        let existing_face_count = self.face_detections.len();
        if self.face_detections.is_empty() {
            if self.config.use_retinaface {
                self.retinaface_detect.detect(frame, &mut self.face_detections)?;
            }

            if self.config.use_internal_face_detection {
                self.detect_faces(frame)?;
            }

            if self.config.assume_fullscreen_face {
                self.face_detections.push((
                    Vec2::ZERO, vec2(self.config.size.x as f32, self.config.size.y as f32)));
            }

            self.wait_count = 0;
        } else if self.face_detections.len() >= self.config.max_faces {
            self.wait_count = 0;
        } else if self.wait_count >= self.config.scan_every {
            if self.config.use_retinaface {
                self.retinaface_scan.detect(frame, &mut self.face_detections)?;
            }

            if self.config.use_internal_face_detection {
                self.detect_faces(frame)?;
            }

            self.wait_count = 0;
        }

        if self.face_detections.is_empty() {
            return Ok(());
        }

        self.num_pending_faces = 0;
        let mut face_boxes = std::mem::take(&mut self.face_detections);
        for (index, (min, size)) in face_boxes.drain(..).enumerate() {
            if self.num_pending_faces >= self.pending_faces.len() {
                break;
            }

            let expand = if index >= existing_face_count { 0.0 } else { 0.1 };
            let mid = min + size * 0.5;
            let half_size = size * (0.6 + expand);
            if half_size.x < 2. || half_size.y < 2. {
                continue;
            }

            let min = mid - half_size;
            let max = mid + half_size;

            let index = self.num_pending_faces;
            let c = self.detect_landmarks(frame, index, min, max)? + expand;
            if c < self.config.threshold {
                continue;
            }

            self.num_pending_faces += 1;
            let ((eye_r, c_r), (eye_l, c_l)) = self.detect_eyes(frame, index)?;

            let face = &mut self.pending_faces[index];
            face.disabled = false;
            face.confidence = c;
            face.landmark_confidence.extend([c_r, c_l]);
            face.landmarks.extend([eye_r, eye_l]);
            face.update_bounds();
        }
        self.face_detections = face_boxes;

        self.pending_face_indices.clear();
        self.pending_face_indices.extend(0..self.num_pending_faces);
        self.pending_face_indices.sort_by(|&idx_a, &idx_b| {
            let a = &self.pending_faces[idx_a];
            let b = &self.pending_faces[idx_b];
            b.confidence.total_cmp(&a.confidence)
        });

        for i in 1..self.num_pending_faces {
            let face = &self.pending_faces[i];
            let min = face.bounds_min;
            let max = face.bounds_max;

            if self.pending_faces[0..i].iter().any(|other| {
                min.cmplt(other.bounds_max).all()
                    && !max.cmpgt(other.bounds_min).any()
            }) {
                self.pending_faces[i].disabled = true;
            }
        }
        self.pending_face_indices.retain(|&idx| !self.pending_faces[idx].disabled);

        let mut num_tracked_faces = 0;
        self.tracked_faces.clear();
        self.tracked_faces.extend(0..self.faces.len());
        while !self.pending_face_indices.is_empty()
            && num_tracked_faces < self.faces.len() {

            let indices = &self.tracked_faces[num_tracked_faces..];
            let mut best = None;

            for (pending_idx, &face_index) in self.pending_face_indices.iter().enumerate() {
                let pending = &self.pending_faces[face_index];

                for (tracked_idx, &face_index) in indices.iter().enumerate() {
                    let face = &self.faces[face_index];
                    let dist2 = if face.is_alive() {
                        (face.centre() - pending.centre).length_squared()
                    } else {
                        f32::MAX
                    };
                    if best.map_or(true, |(d2, _, _)| dist2 < d2) {
                        best = Some((dist2, pending_idx, tracked_idx));
                    }
                }
            }

            let (_, pending_idx, tracked_idx) = best.unwrap();
            let idx = self.pending_face_indices.swap_remove(pending_idx);
            let face_index = self.tracked_faces[num_tracked_faces + tracked_idx];
            self.tracked_faces.swap(num_tracked_faces, num_tracked_faces + tracked_idx);
            num_tracked_faces += 1;

            let pending = &mut self.pending_faces[idx];
            let tracked = &mut self.faces[face_index];
            let centre = pending.centre;
            let size = pending.size;
            self.face_detections.push((centre - size * 0.5, size));

            tracked.update(
                frame.width(),
                frame.height(),
                pending.image_min,
                pending.image_max,
                pending.centre,
                pending.size,
                &mut pending.landmark_confidence,
                &mut pending.landmarks,
            );

        }

        for index in self.tracked_faces.drain(num_tracked_faces..) {
            let face = &mut self.faces[index];
            face.reset();
        }

        Ok(())
    }
}
