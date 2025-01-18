use glam::Vec2;

pub fn align_points(a: Vec2, b: Vec2, points: &mut [Vec2]) -> f32 {
    let angle = (b - a).to_angle();
    let rot = Vec2::from_angle(-angle);
    for point in points {
        *point = a + rot.rotate(*point - a);
    }
    angle
}
