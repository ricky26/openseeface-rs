use std::ops::Deref;
use image::{ImageBuffer, Pixel};
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
