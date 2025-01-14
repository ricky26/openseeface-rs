use std::ops::{Deref, DerefMut};
use image::{ImageBuffer, Pixel, Rgba};

pub(crate) trait FromColorValue<T> {
    fn from_color_value(value: T) -> Self;
}

impl FromColorValue<u8> for u8 {
    fn from_color_value(value: u8) -> Self {
        value
    }
}

impl FromColorValue<f32> for f32 {
    fn from_color_value(value: f32) -> Self {
        value
    }
}

impl FromColorValue<f32> for u8 {
    fn from_color_value(value: f32) -> Self {
        (value.clamp(0., 1.) * (u8::MAX as f32)) as u8
    }
}

impl FromColorValue<u8> for f32 {
    fn from_color_value(value: u8) -> Self {
        (value as f32) / (u8::MAX as f32)
    }
}

fn to_rgba<P: Pixel, F>(p: &P) -> Rgba<F>
    where F: FromColorValue<<P as Pixel>::Subpixel>
{
    let rgba = p.to_rgba();
    let r = F::from_color_value(rgba[0]);
    let g = F::from_color_value(rgba[1]);
    let b = F::from_color_value(rgba[2]);
    let a = F::from_color_value(rgba[3]);
    Rgba([r, g, b, a])
}

fn lerp_rgba_f32(a: Rgba<f32>, b: Rgba<f32>, x: f32) -> Rgba<f32> {
    let y = 1. - x;
    let [r0, g0, b0, a0] = a.0;
    let [r1, g1, b1, a1] = b.0;
    Rgba([r0 * y + r1 * x, g0 * y + g1 * x, b0 * y + b1 * x, a0 * y + a1 * x])
}

pub(crate) fn resize_linear_rgba<P1, C1, P2, C2>(src: &ImageBuffer<P1, C1>, dest: &mut ImageBuffer<P2, C2>)
where
    P1: Pixel,
    f32: FromColorValue<<P1 as Pixel>::Subpixel>,
    C1: Deref<Target=[P1::Subpixel]>,
    P2: Pixel,
    <P2 as Pixel>::Subpixel: FromColorValue<f32>,
    C2: Deref<Target=[P2::Subpixel]> + DerefMut,
{
    let src_w = src.width();
    let src_h = src.height();
    let scale_x = (src_w as f32) / (dest.width() as f32);
    let scale_y = (src_h as f32) / (dest.height() as f32);
    let max_x = src_w.saturating_sub(2);
    let max_y = src_h.saturating_sub(2);

    for (x, y, pixel) in dest.enumerate_pixels_mut() {
        let x_f = scale_x * (x as f32);
        let y_f = scale_y * (y as f32);

        let src_x = (x_f as u32).min(max_x);
        let src_y = (y_f as u32).min(max_y);

        let mix_x = (x_f - (src_x as f32)).clamp(0., 1.);
        let mix_y = (y_f - (src_y as f32)).clamp(0., 1.);

        let a = to_rgba(src.get_pixel(src_x, src_y));
        let b = to_rgba(src.get_pixel(src_x + 1, src_y));
        let c = to_rgba(src.get_pixel(src_x, src_y + 1));
        let d = to_rgba(src.get_pixel(src_x + 1, src_y + 1));

        let e = lerp_rgba_f32(a, b, mix_x);
        let f = lerp_rgba_f32(c, d, mix_x);
        let g = lerp_rgba_f32(e, f, mix_y);

        let channels = pixel.channels_mut();
        channels[0] = FromColorValue::from_color_value(g[0]);
        channels[1] = FromColorValue::from_color_value(g[1]);
        channels[2] = FromColorValue::from_color_value(g[2]);
        channels[3] = FromColorValue::from_color_value(g[3]);
    }
}
