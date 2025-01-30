use std::borrow::Borrow;
use std::marker::PhantomData;

use bevy::color::palettes::css::{DARK_GREEN, GREEN, WHITE};
use bevy::input::common_conditions::input_just_pressed;
use bevy::prelude::*;

use crate::ActiveTracker;

trait FeatureSet: 'static {
    type Features: Borrow<[f32]>;
    const FEATURE_NAMES: &'static [&'static str];

    fn values_from_tracker(tracker: &ActiveTracker, index: usize) -> Self::Features;
}

struct OsfFeatures;

impl FeatureSet for OsfFeatures {
    type Features = [f32; 14];
    const FEATURE_NAMES: &'static [&'static str] = &[
        "eye_l",
        "eye_r",
        "eyebrow_updown_l",
        "eyebrow_updown_r",
        "eyebrow_quirk_l",
        "eyebrow_quirk_r",
        "eyebrow_steepness_l",
        "eyebrow_steepness_r",
        "mouth_corner_updown_l",
        "mouth_corner_updown_r",
        "mouth_corner_inout_l",
        "mouth_corner_inout_r",
        "mouth_open",
        "mouth_wide",
    ];

    fn values_from_tracker(tracker: &ActiveTracker, index: usize) -> Self::Features {
        let features = tracker.osf_features.current_features();
        let features = &features[index];
        [
            features.eye_l,
            features.eye_r,
            features.eyebrow_updown_l,
            features.eyebrow_updown_r,
            features.eyebrow_quirk_l,
            features.eyebrow_quirk_r,
            features.eyebrow_steepness_l,
            features.eyebrow_steepness_r,
            features.mouth_corner_updown_l,
            features.mouth_corner_updown_r,
            features.mouth_corner_inout_l,
            features.mouth_corner_inout_r,
            features.mouth_open,
            features.mouth_wide,
        ]
    }
}

struct ArkitFeaturesA;

impl FeatureSet for ArkitFeaturesA {
    type Features = [f32; 26];
    const FEATURE_NAMES: &'static [&'static str] = &[
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
    ];

    fn values_from_tracker(tracker: &ActiveTracker, index: usize) -> Self::Features {
        let features = tracker.arkit_features.current_features();
        let features = &features[index];
        [
            features.brow_down_left,
            features.brow_down_right,
            features.brow_inner_up,
            features.brow_outer_up_left,
            features.brow_outer_up_right,
            features.cheek_puff,
            features.cheek_squint_left,
            features.cheek_squint_right,
            features.eye_blink_left,
            features.eye_blink_right,
            features.eye_look_down_left,
            features.eye_look_down_right,
            features.eye_look_in_left,
            features.eye_look_in_right,
            features.eye_look_out_left,
            features.eye_look_out_right,
            features.eye_look_up_left,
            features.eye_look_up_right,
            features.eye_squint_left,
            features.eye_squint_right,
            features.eye_wide_left,
            features.eye_wide_right,
            features.jaw_forward,
            features.jaw_left,
            features.jaw_open,
            features.jaw_right,
        ]
    }
}

struct ArkitFeaturesB;

impl FeatureSet for ArkitFeaturesB {
    type Features = [f32; 25];
    const FEATURE_NAMES: &'static [&'static str] = &[
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "noseSneerLeft",
        "noseSneerRight",
    ];

    fn values_from_tracker(tracker: &ActiveTracker, index: usize) -> Self::Features {
        let features = tracker.arkit_features.current_features();
        let features = &features[index];
        [
            features.mouth_close,
            features.mouth_dimple_left,
            features.mouth_dimple_right,
            features.mouth_frown_left,
            features.mouth_frown_right,
            features.mouth_funnel,
            features.mouth_left,
            features.mouth_lower_down_left,
            features.mouth_lower_down_right,
            features.mouth_press_left,
            features.mouth_press_right,
            features.mouth_pucker,
            features.mouth_right,
            features.mouth_roll_lower,
            features.mouth_roll_upper,
            features.mouth_shrug_lower,
            features.mouth_shrug_upper,
            features.mouth_smile_left,
            features.mouth_smile_right,
            features.mouth_stretch_left,
            features.mouth_stretch_right,
            features.mouth_upper_up_left,
            features.mouth_upper_up_right,
            features.nose_sneer_left,
            features.nose_sneer_right,
        ]
    }
}

#[derive(Clone, Debug, Default, Component)]
struct FeaturesUi<F> {
    features: Vec<Entity>,
    _phantom: PhantomData<fn(&F)>,
}

#[derive(Clone, Debug, Default, Component)]
struct FeatureElement;

fn spawn_ui<F: FeatureSet>(mut commands: Commands) {
    let mut features = Vec::with_capacity(F::FEATURE_NAMES.len());
    commands
        .spawn((
            Name::new("FeaturesUi"),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.),
                right: Val::Px(0.),
                ..default()
            },
            Visibility::Hidden,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Name::new("Labels"),
                    Node {
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(Val::Px(5.)),
                        row_gap: Val::Px(5.),
                        ..default()
                    },
                ))
                .with_children(|parent| {
                    for &name in F::FEATURE_NAMES {
                        parent.spawn((
                            Name::new(name),
                            Node {
                                height: Val::Px(16.),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                ..default()
                            },
                            Text::new(name),
                            TextFont {
                                font_size: 16.,
                                ..default()
                            },
                            Label,
                        ));
                    }
                });

            parent
                .spawn((
                    Name::new("Bars"),
                    Node {
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(Val::Px(5.)),
                        row_gap: Val::Px(5.),
                        ..default()
                    },
                ))
                .with_children(|parent| {
                    for &name in F::FEATURE_NAMES {
                        parent
                            .spawn((
                                Name::new(name),
                                Node {
                                    width: Val::Px(80.),
                                    height: Val::Px(14.),
                                    margin: UiRect::all(Val::Px(1.)),
                                    border: UiRect::all(Val::Px(1.)),
                                    ..default()
                                },
                                BorderColor(WHITE.into()),
                                BackgroundColor(DARK_GREEN.into()),
                            ))
                            .with_children(|parent| {
                                let element = parent
                                    .spawn((
                                        Name::new(name),
                                        Node {
                                            width: Val::Percent(0.),
                                            height: Val::Percent(100.),
                                            ..default()
                                        },
                                        BackgroundColor(GREEN.into()),
                                        FeatureElement,
                                    ))
                                    .id();
                                features.push(element);
                            });
                    }
                });
        })
        .insert(FeaturesUi::<F> {
            features,
            _phantom: PhantomData,
        });
}

fn update_ui<F: FeatureSet>(
    tracker: Res<ActiveTracker>,
    feature_ui: Single<&FeaturesUi<F>>,
    mut nodes: Query<&mut Node, With<FeatureElement>>,
) {
    let Some(index) = tracker.tracker.faces().iter().rposition(|f| f.is_alive()) else {
        return;
    };

    let features = F::values_from_tracker(&tracker, index);
    for (&entity, &value) in feature_ui.features.iter().zip(features.borrow()) {
        let Ok(mut node) = nodes.get_mut(entity) else {
            continue;
        };

        let percent = (value + 1.) / 2. * 100.;
        node.width = Val::Percent(percent);
    }
}

fn toggle_ui<F: FeatureSet>(mut feature_ui: Single<&mut Visibility, With<FeaturesUi<F>>>) {
    feature_ui.toggle_inherited_hidden();
}

struct FeaturesPlugin<F, C> {
    toggle_condition: C,
    _phantom: PhantomData<fn(&F)>,
}

impl<F: FeatureSet, C> FeaturesPlugin<F, C> {
    pub fn new<M>(toggle_condition: C) -> FeaturesPlugin<F, <C as IntoSystem<(), bool, M>>::System>
        where C: Condition<M>
    {
        FeaturesPlugin {
            toggle_condition: IntoSystem::into_system(toggle_condition),
            _phantom: PhantomData,
        }
    }
}

impl<C: Condition<()> + Send + Sync + Clone + 'static, F: FeatureSet> Plugin for FeaturesPlugin<F, C> {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, spawn_ui::<F>)
            .add_systems(Update, (
                update_ui::<F>,
                toggle_ui::<F>.run_if(self.toggle_condition.clone()),
            ));
    }
}

pub fn plugin(app: &mut App) {
    app
        .add_plugins((
            FeaturesPlugin::<OsfFeatures, _>::new(input_just_pressed(KeyCode::F5)),
            FeaturesPlugin::<ArkitFeaturesA, _>::new(input_just_pressed(KeyCode::BracketLeft)),
            FeaturesPlugin::<ArkitFeaturesB, _>::new(input_just_pressed(KeyCode::BracketRight)),
        ));
}
