use crate::face::TrackedFace;

pub mod openseeface;

pub trait FeatureExtractor {
    type Features: Default;

    type Config;

    fn from_config(config: Self::Config) -> Self;

    fn update(&mut self, features: &mut Self::Features, face: &TrackedFace, now: f64);
}

#[derive(Clone, Debug, Default)]
pub struct FeatureTracker<E: FeatureExtractor> {
    feature_extractors: Vec<E>,
    current_features: Vec<E::Features>,
}

impl<E: FeatureExtractor> FeatureTracker<E> {
    pub fn current_features(&self) -> &[E::Features] {
        &self.current_features
    }

    pub fn from_iter(config: impl Iterator<Item = E::Config>) -> FeatureTracker<E> {
        let (min, _) = config.size_hint();
        let mut feature_extractors = Vec::with_capacity(min);
        let mut current_features = Vec::with_capacity(min);
        for config in config {
            feature_extractors.push(E::from_config(config));
            current_features.push(E::Features::default());
        }
        FeatureTracker {
            feature_extractors,
            current_features,
        }
    }

    pub fn update(&mut self, faces: &[TrackedFace], now: f64) {
        for ((face, extractor), features) in faces.iter()
            .zip(&mut self.feature_extractors)
            .zip(&mut self.current_features) {

            if face.has_pose() {
                extractor.update(features, face, now);
            }
        }
    }
}

impl<E: FeatureExtractor> FeatureTracker<E> where E: Default {
    pub fn new(max_faces: usize) -> FeatureTracker<E> {
        let mut feature_extractors = Vec::new();
        feature_extractors.resize_with(max_faces, E::default);

        let mut current_features = Vec::new();
        current_features.resize_with(max_faces, E::Features::default);

        FeatureTracker {
            feature_extractors,
            current_features,
        }
    }
}

impl<E: FeatureExtractor> FeatureTracker<E> where E::Config: Default {
    pub fn from_default_config(max_faces: usize) -> FeatureTracker<E> {
        let mut feature_extractors = Vec::new();
        feature_extractors.resize_with(max_faces, || E::from_config(E::Config::default()));

        let mut current_features = Vec::new();
        current_features.resize_with(max_faces, E::Features::default);

        FeatureTracker {
            feature_extractors,
            current_features,
        }
    }
}

impl<E: FeatureExtractor> FeatureTracker<E> where E::Config: Clone {
    pub fn from_config(max_faces: usize, config: E::Config) -> FeatureTracker<E> {
        let mut feature_extractors = Vec::new();
        feature_extractors.resize_with(max_faces, || E::from_config(config.clone()));

        let mut current_features = Vec::new();
        current_features.resize_with(max_faces, E::Features::default);

        FeatureTracker {
            feature_extractors,
            current_features,
        }
    }
}
