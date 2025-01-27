use bevy::color::palettes::css::{DARK_GREEN, GREEN, WHITE};
use bevy::input::common_conditions::input_just_pressed;
use bevy::prelude::*;
use crate::ActiveTracker;

const FEATURE_NAMES: [&'static str; 14] = [
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

#[derive(Clone, Debug, Default, Component)]
struct FeaturesUi {
    features: Vec<Entity>,
}

#[derive(Clone, Debug, Default, Component)]
struct FeatureElement;

fn spawn_ui(mut commands: Commands) {
    let mut features = Vec::with_capacity(FEATURE_NAMES.len());
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
                    for name in FEATURE_NAMES {
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
                    for name in FEATURE_NAMES {
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
        .insert(FeaturesUi {
            features,
        });
}

fn update_ui(
    tracker: Res<ActiveTracker>,
    feature_ui: Single<&FeaturesUi>,
    mut nodes: Query<&mut Node, With<FeatureElement>>,
) {
    let Some(index) = tracker.tracker.faces().iter().rposition(|f| f.is_alive()) else {
        return;
    };

    let features = tracker.features.current_features();
    let features = &features[index];
    let feature_list = [
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
    ];

    for (&entity, &value) in feature_ui.features.iter().zip(&feature_list) {
        let Ok(mut node) = nodes.get_mut(entity) else {
            continue;
        };

        let percent = (value + 1.) / 2. * 100.;
        node.width = Val::Percent(percent);
    }
}

fn toggle_ui(mut feature_ui: Single<&mut Visibility, With<FeaturesUi>>) {
    feature_ui.toggle_inherited_hidden();
}

pub fn plugin(app: &mut App) {
    app
        .add_systems(Startup, spawn_ui)
        .add_systems(Update, (
            update_ui,
            toggle_ui.run_if(input_just_pressed(KeyCode::F5)),
        ));
}
