use std::sync::LazyLock;

use glam::{uvec2, vec2, vec3, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use image::{ImageBuffer, Rgb, Rgb32FImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use ndarray::Axis;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::image::ImageArrayExt;

const DEFAULT_MODEL: &[u8] = include_bytes!("../models/retinaface_640x640_opt.onnx");
const DEFAULT_JSON: &[u8] = include_bytes!("../models/priorbox_640x640.json");

static PRIORS: LazyLock<Vec<Vec4>> = LazyLock::new(|| {
    serde_json::from_slice(DEFAULT_JSON).expect("default retinaface JSON should be valid")
});

fn nms(faces: &[(Vec2, Vec2)], threshold: f32) {
}

pub struct RetinaFaceDetector {
    session: Session,
    resolution: UVec2,
    priors: &'static Vec<Vec4>,
    variances: Vec2,
    min_confidence: f32,
    nms_threshold: f32,
    top_k: usize,
    scratch_image: Rgb32FImage,
    nms_keep: Vec<usize>,
}

impl RetinaFaceDetector {
    pub fn new(
        num_threads: usize,
        min_confidence: f32,
        nms_threshold: f32,
        top_k: usize,
    ) -> Result<RetinaFaceDetector, ort::Error> {
        let resolution = uvec2(640, 640);
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(num_threads)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(DEFAULT_MODEL)?;
        let scratch_image = ImageBuffer::new(resolution.x, resolution.y);
        Ok(Self {
            session,
            resolution,
            priors: &PRIORS,
            variances: vec2(0.1, 0.2),
            min_confidence,
            nms_threshold,
            top_k,
            scratch_image,
            nms_keep: Vec::new(),
        })
    }

    pub fn detect(
        &mut self, frame: &Rgb32FImage, faces: &mut Vec<(Vec2, Vec2)>,
    ) -> Result<(), ort::Error> {
        let mean = vec3(0.407843, 0.458824, 0.482353);
        let frame_size = uvec2(frame.width(), frame.height());
        let scale = self.resolution.as_vec2() / frame_size.as_vec2();
        let scale = scale.min_element();
        let used = frame_size.as_vec2() * scale;
        let offset = (frame_size.as_vec2() - used) * 0.5;
        let projection = Projection::translate(offset.x, offset.y)
            * Projection::scale(scale, scale);
        warp_into(
            frame,
            &projection,
            Interpolation::Bilinear,
            Rgb(mean.to_array()),
            &mut self.scratch_image,
        );

        for p in self.scratch_image.pixels_mut() {
            p.0 = ((Vec3::from_array(p.0) - mean) * 255.).to_array();
        }

        let input = self.scratch_image.as_view()
            .insert_axis(Axis(0))
            .permuted_axes((0, 3, 1, 2));

        let outputs = self.session.run(ort::inputs!{
            "input0" => input,
        }?)?;

        let loc = outputs[0].try_extract_tensor::<f32>()?;
        let conf = outputs[1].try_extract_tensor::<f32>()?;

        let faces_offset = faces.len();
        let n = loc.shape()[1];
        for i in 0..n {
            let c = conf[[0, i, 1]];
            if c <= self.min_confidence {
                continue;
            }

            let prior = self.priors[i];
            let mid = vec2(loc[[0, i, 0]], loc[[0, i, 1]]);
            let size = vec2(loc[[0, i, 2]], loc[[0, i, 3]]);
            let mid = prior.xy() + mid * self.variances.x * prior.zw();
            let size = prior.zw() * (size * self.variances[1]).exp();

            let half_size = size * 0.5;
            let min = mid - half_size;
            let max = mid + half_size;
            faces.push((min, max));
        }

        println!("shape loc={:?} conf={:?}", loc.shape(), conf.shape());

        Ok(())
    }
}
