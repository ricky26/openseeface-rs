use glam::Vec2;

pub fn angle(p1: Vec2, p2: Vec2) -> f32 {
    let delta = p2 - p1;
    delta.y.atan2(delta.x)
}

pub fn align_points(a: Vec2, b: Vec2, points: &mut [Vec2]) -> f32 {
    let alpha = angle(a, b);
    let rot = Vec2::from_angle(-alpha);
    for point in points {
        *point = a + rot.rotate(*point - a);
    }
    alpha
}
