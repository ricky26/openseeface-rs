use std::ops::Deref;
use image::{ImageBuffer, Pixel, Rgb32FImage, RgbImage, RgbaImage};
use ndarray::{ArrayView, Dimension, Ix3};

pub(crate) trait ImageArrayExt {
    type Dimension: Dimension;

    fn as_view(&self) -> ArrayView<f32, Self::Dimension>;
}

impl<P, C> ImageArrayExt for ImageBuffer<P, C>
    where P: Pixel<Subpixel = f32>,
    C: Deref<Target = [f32]>
{
    type Dimension = Ix3;

    fn as_view(&self) -> ArrayView<f32, Ix3> {
        let slice = &**self.as_raw();
        let w = self.width() as usize;
        let h = self.height() as usize;
        let shape = (h, w, P::CHANNEL_COUNT as usize);
        ArrayView::from_shape(shape, slice).unwrap()
    }
}

// Specialised conversion functions for performance.

/// Convert RGB8 to RGB32F image with existing image.
pub fn rgb_to_rgb32f(dest: &mut Rgb32FImage, src: &RgbImage) {
    if dest.dimensions() != src.dimensions() {
        *dest = Rgb32FImage::new(src.width(), src.height());
    }

    for (dest, src) in dest.chunks_exact_mut(3)
        .zip(src.chunks_exact(3)) {
        dest[0] = (src[0] as f32) / 255.;
        dest[1] = (src[1] as f32) / 255.;
        dest[2] = (src[2] as f32) / 255.;
    }
}

/// Convert RGB8 to RGBA8 image with existing image.
pub fn rgb_to_rgba(dest: &mut RgbaImage, src: &RgbImage) {
    if dest.dimensions() != src.dimensions() {
        *dest = RgbaImage::new(src.width(), src.height());
    }

    for (dest, src) in dest.chunks_exact_mut(4)
        .zip(src.chunks_exact(3)) {
        dest[0] = src[0];
        dest[1] = src[1];
        dest[2] = src[2];
        dest[3] = 255;
    }
}
