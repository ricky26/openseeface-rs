use std::sync::LazyLock;

use glam::{ivec2, IVec2, Vec2, Vec4};
use image::Rgb32FImage;
use ndarray::Array1;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

const DEFAULT_MODEL: &[u8] = include_bytes!("../models/retinaface_640x640_opt.onnx");
const DEFAULT_JSON: &[u8] = include_bytes!("../models/priorbox_640x640.json");

static PRIORS: LazyLock<Vec<Vec4>> = LazyLock::new(|| {
    serde_json::from_slice(DEFAULT_JSON).expect("default retinaface JSON should be valid")
});

pub struct RetinaFaceDetector {
    session: Session,
    resolution: IVec2,
    priors: &'static Vec<Vec4>,
    min_confidence: f32,
    nms_threshold: f32,
    top_k: usize,
}

impl RetinaFaceDetector {
    pub fn new(
        num_threads: usize,
        min_confidence: f32,
        nms_threshold: f32,
        top_k: usize,
    ) -> Result<RetinaFaceDetector, ort::Error> {
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(num_threads)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(DEFAULT_MODEL)?;
        Ok(Self {
            session,
            resolution: ivec2(640, 480),
            priors: &PRIORS,
            min_confidence,
            nms_threshold,
            top_k,
        })
    }

    pub fn detect(
        &self, frame: &Rgb32FImage, faces: &mut Vec<(Vec2, Vec2)>,
    ) -> Result<(), ort::Error> {
        let outputs = self.session.run(ort::inputs!{
            "input0" => Array1::<f32>::zeros(0),
        }?)?;

        for (output, _) in &outputs {
            println!("output {output}");
        }

        outputs.get("output0")
            .expect("output0 should exist");

        outputs.get("output1")
            .expect("output1 should exist");

        Ok(())
    }
}
