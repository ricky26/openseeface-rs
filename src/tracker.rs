use glam::{uvec2, vec3, vec4, UVec2, Vec3, Vec4};
use image::{Rgb32FImage, RgbImage};
use ndarray::{s, ArrayView, ShapeBuilder};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::face::{TrackedFace, DEFAULT_FACE};
use crate::image::resize_linear_rgb;
use crate::retinaface::RetinaFaceDetector;

const MODEL_0: &'static [u8] = include_bytes!("../models/lm_model0_opt.onnx");
const MODEL_1: &'static [u8] = include_bytes!("../models/lm_model1_opt.onnx");
const MODEL_2: &'static [u8] = include_bytes!("../models/lm_model2_opt.onnx");
const MODEL_3: &'static [u8] = include_bytes!("../models/lm_model3_opt.onnx");
const MODEL_4: &'static [u8] = include_bytes!("../models/lm_model4_opt.onnx");
const MODEL_T: &'static [u8] = include_bytes!("../models/lm_modelT_opt.onnx");
const MODEL_V: &'static [u8] = include_bytes!("../models/lm_modelV_opt.onnx");
const MODEL_U: &'static [u8] = include_bytes!("../models/lm_modelU_opt.onnx");

const MODEL_GAZE: &'static [u8] = include_bytes!("../models/mnv3_gaze32_split_opt.onnx");

const MODEL_DETECTION: &'static [u8] = include_bytes!("../models/mnv3_detection_opt.onnx");

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum TrackerModel {
    Model0,
    Model1,
    Model2,
    Model3,
    Model4,
    ModelT,
    ModelV,
    ModelU,
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

const CONTOUR_INDICES: [usize; 14] = [0,1,8,15,16,27,28,29,30,31,32,33,34,35];
const CONTOUR_INDICES_T: [usize; 8] = [0,2,8,14,16,27,30,33];

#[derive(Clone, Debug)]
pub struct TrackerConfig {
    pub size: UVec2,
    pub model_type: TrackerModel,
    pub max_faces: usize,
    pub detection_threshold: f32,
    pub threshold: Option<f32>,
    pub discard_after: usize,
    pub scan_every: usize,
    pub bbox_growth: f32,
    pub max_threads: usize,
    pub no_gaze: bool,
    pub use_retina_face: bool,
    pub use_internal_face_detection: bool,
    pub assume_fullscreen_face: bool,
    pub max_feature_updates: f64,
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
            threshold: None,
            discard_after: 5,
            scan_every: 3,
            bbox_growth: 0.0,
            max_threads: 4,
            no_gaze: false,
            use_retina_face: false,
            use_internal_face_detection: true,
            assume_fullscreen_face: false,
            max_feature_updates: 0.0,
            static_model: false,
            feature_level: 2,
        }
    }
}

pub struct Tracker {
    config: TrackerConfig,
    face_3d: [Vec3; 70],
    retina_face_detect: RetinaFaceDetector,
    retina_face_scan: RetinaFaceDetector,
    session: Session,
    sessions: Vec<Session>,
    gaze_session: Session,
    face_detect: Session,
    face_detect_224: Rgb32FImage,
    mean: Vec3,
    std: Vec3,
    frame_count: usize,
    tracked_faces: Vec<TrackedFace>,
    wait_count: usize,
    face_boxes: Vec<Vec4>,
}

impl Tracker {
    pub fn face_boxes(&self) -> &[Vec4] {
        &self.face_boxes
    }

    pub fn new(
        config: TrackerConfig,
    ) -> Result<Tracker, ort::Error> {
        let face_detector = RetinaFaceDetector::new(
            config.max_threads.max(4),
            0.4,
            0.4,
            config.max_faces,
        )?;
        let face_detector_scan = RetinaFaceDetector::new(
            2,
            0.4,
            0.4,
            config.max_faces,
        )?;

        let model_data = model_data(config.model_type);
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(config.max_threads.min(4))?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(model_data)?;

        let max_workers = config.max_threads.min(config.max_faces).max(1);
        let threads_per_worker = (config.max_threads / max_workers).max(1);
        let extra_threads = config.max_threads % max_workers;
        let mut sessions = Vec::with_capacity(max_workers);
        for i in 0..max_workers {
            let num_threads = threads_per_worker + if i < extra_threads { 1 } else { 0 };
            let session = Session::builder()?
                .with_inter_threads(1)?
                .with_intra_threads(num_threads)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_memory(model_data)?;
            sessions.push(session);
        }

        let gaze_session = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(1)?
            .commit_from_memory(MODEL_GAZE)?;
        let gaze_detection = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(1)?
            .commit_from_memory(MODEL_DETECTION)?;

        let mean = vec3(0.485, 0.456, 0.406);
        let std = vec3(0.229, 0.224, 0.225);
        let mean = mean / std;
        let std = 1. / std;

        let face_3d = DEFAULT_FACE;
        let mut tracked_faces = Vec::with_capacity(config.max_faces);
        let contour_indices = if config.model_type == TrackerModel::ModelT { &CONTOUR_INDICES_T[..] } else { &CONTOUR_INDICES };

        for i in 0..config.max_faces {
            tracked_faces.push(TrackedFace::new(i, contour_indices));
        }

        let face_detect_224 = Rgb32FImage::new(224, 224);

        Ok(Self {
            config,
            face_3d,
            retina_face_detect: face_detector,
            retina_face_scan: face_detector_scan,
            session,
            sessions,
            gaze_session,
            face_detect: gaze_detection,
            face_detect_224,
            mean,
            std,
            frame_count: 0,
            tracked_faces,
            wait_count: 0,
            face_boxes: Vec::new(),
        })
    }

    #[tracing::instrument(skip_all)]
    fn detect_faces(&mut self, frame: &RgbImage) -> Result<(), ort::Error> {
        resize_linear_rgb(frame, &mut self.face_detect_224);
        for pixel in self.face_detect_224.pixels_mut() {
            pixel.0 = (Vec3::from_array(pixel.0) * self.std - self.mean).to_array();
        }

        tracing::info!("std={} mean={}", self.std, self.mean);

        let input = ArrayView::from_shape((1, 224, 224, 3), self.face_detect_224.as_raw())
            .unwrap()
            .permuted_axes((0, 3, 1, 2));
        tracing::info!("input strides={:?} {input:?}", input.strides());
        let output = self.face_detect.run(ort::inputs![input]?)?;
        let max_pool = output["maxpool"].try_extract_tensor::<f32>()?;
        let output = output["output"].try_extract_tensor::<f32>()?;

        tracing::info!("output {output:?}");
        tracing::info!("max_pool {max_pool:?}");

        let scale_x = (frame.width() as f32) / 224.;
        let scale_y = (frame.height() as f32) / 224.;

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
                let x = (x as f32 * 4. - r) * scale_x;
                let y = (y as f32 * 4. - r) * scale_y;
                let w = (2. * r) * scale_x;
                let h = (2. * r) * scale_y;
                tracing::info!("face {x},{y} {w},{h}");
                self.face_boxes.push(vec4(x, y, w, h));
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn detect(&mut self, frame: &RgbImage, now: f64) -> Result<(), ort::Error> {
        // TODO: remove this
        self.face_boxes.clear();

        self.frame_count += 1;
        self.wait_count += 1;

        if self.face_boxes.is_empty() {
            if self.config.use_retina_face {
                self.retina_face_detect.detect(frame, &mut self.face_boxes)?;
            }

            if self.config.use_internal_face_detection {
                self.detect_faces(frame)?;
            }

            if self.config.assume_fullscreen_face {
                self.face_boxes.push(vec4(0., 0., self.config.size.x as f32, self.config.size.y as f32));
            }

            self.wait_count = 0;
        } else if self.face_boxes.len() >= self.config.max_faces {
            self.wait_count = 0;
        } else if self.wait_count >= self.config.scan_every {
            if self.config.use_retina_face {
                self.retina_face_scan.detect(frame, &mut self.face_boxes)?;
            }

            if self.config.use_internal_face_detection {
                self.detect_faces(frame)?;
            }

            self.wait_count = 0;
        }

        if self.face_boxes.is_empty() {
            // ...
            return Ok(());
        }

        Ok(())
    }
}
