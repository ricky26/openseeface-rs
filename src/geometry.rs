use std::f32::consts::PI;
use glam::Vec2;

pub fn normalise_angle(mut angle: f32) -> f32 {
    while angle < -PI / 2. {
        angle += PI;
    }

    while angle > PI / 2. {
        angle -= PI;
    }

    angle
}

pub fn rotate(origin: Vec2, point: Vec2, a: f32) -> Vec2 {
    origin + Vec2::from_angle(-a).rotate(point - origin)
}

pub fn angle(p1: Vec2, p2: Vec2) -> f32 {
    let delta = p2 - p1;
    delta.y.atan2(delta.x)
}

pub fn compensate(p1: Vec2, p2: Vec2) -> (Vec2, f32) {
    let angle = angle(p1, p2);
    let p3 = rotate(p1, p2, angle);
    (p3, angle)
}

pub fn align_points(a: Vec2, b: Vec2, points: &mut [Vec2]) -> f32 {
    let alpha = normalise_angle(angle(a, b));
    for point in points {
        *point = rotate(a, *point, alpha);
    }
    alpha
}
