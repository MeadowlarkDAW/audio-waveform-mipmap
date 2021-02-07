use std::cmp::{max, min};
use std::vec::Vec;

pub struct DisplayConfig {
    preamp: f32,
    mode: DisplayMode,
}

pub enum DisplayMode {
    Linear,
    DB(f32),
}

impl DisplayConfig {
    #[inline]
    // Generic over value type?
    pub fn convert_sample(&self, value: f32) -> i8 {
        let value = value * self.preamp;
        let value = match self.mode {
            DisplayMode::Linear => value.max(-1.).min(1.),
            DisplayMode::DB(lowest) => {
                let db_value = 10. * value.abs().log(10.);
                let scaled = ((db_value / lowest) + 1.).max(0.).min(1.);
                scaled.copysign(value)
            }
        };
        (value * std::i8::MAX as f32) as i8
    }
}

pub const BASE: usize = 4;
pub struct SampleMipMap {
    size: usize,
    config: DisplayConfig,
    tree: Vec<(i8, i8)>,
    layers: Vec<usize>,
}
impl SampleMipMap {
    pub fn new(data: &[f32], config: DisplayConfig) -> Self {
        let size = data.len();
        // + 64 to accomodate for rounding up. Actual upper bound is around log_BASE(size).
        let mut tree = Vec::with_capacity(size / (BASE - 1) + 64);
        let mut layers = Vec::new();
        layers.push(0); // Bottom layer - raw samples
        let mut current_layer = size;
        let mut pos = 0;
        while current_layer >= BASE {
            current_layer = (current_layer + BASE - 1) / BASE;
            if layers.len() <= 1 {
                for i in 0..current_layer {
                    tree.push(
                        data[i * BASE..]
                            .iter()
                            .take(BASE)
                            .map(|v| config.convert_sample(*v))
                            .fold((std::i8::MAX, std::i8::MIN), |a, b| {
                                (min(a.0, b), max(a.1, b))
                            }),
                    );
                }
            } else {
                for i in 0..current_layer {
                    tree.push(
                        tree[layers.last().unwrap() + i * BASE..]
                            .iter()
                            .take(BASE)
                            .fold((std::i8::MAX, std::i8::MIN), |a, b| {
                                (min(a.0, b.1), max(a.1, b.1))
                            }),
                    );
                }
            }
            layers.push(pos);
            pos += current_layer;
        }
        SampleMipMap {
            size,
            config,
            tree,
            layers,
        }
    }

    /// Query a range of data.
    ///
    /// Should produce at most around `pixels * BASE` points (if available).
    ///
    /// Returns a tuple with two elements. The first describes the x-coordinates
    /// of the leftmost and rightmost points of data returned *in pixel space*.
    /// Bear in mind that these can be outside of the viewport.
    ///
    /// The second element of the tuple is an iterator over the `(min, max)`
    /// values obtained.
    ///
    /// # Arguments
    ///
    /// * `data` - The same data that was provided on construction.
    /// * `left` - The leftmost point of the viewport (in samples).
    /// * `width` - The width of the viewport in samples.
    /// * `pixels` - The width of the viewport in pixels.
    ///
    pub fn query<'a>(
        &'a self,
        data: &'a [f32],
        left: f64,
        width: f64,
        pixels: usize,
    ) -> ((f64, f64), impl ExactSizeIterator<Item = (i8, i8)> + 'a) {
        assert!(data.len() == self.size);
        
        let padding = width / (pixels as f64);
        let mut interval = (
            ((left - padding).ceil() as usize).max(0).min(self.size - 1),
            ((left + width + padding).floor() as usize).max(0).min(self.size - 1),
        );
        let mut layer = 0;
        let mut layer_piece_size = 1f64;
        let mut query_width = width;
        while query_width / (BASE as f64) >= pixels as f64 && layer + 1 < self.layers.len() {
            interval = (interval.0 / BASE, interval.1 / BASE);
            layer += 1;
            layer_piece_size *= BASE as f64;
            query_width /= BASE as f64;
        }
        let layer_index_to_px = |i| {
            let x = (i as f64 + 0.5) * layer_piece_size;
            (x - left) / width * (pixels as f64)
        };
        (
            (layer_index_to_px(interval.0), layer_index_to_px(interval.1)),
            // RangeInclusive<usize>: !ExactSizeIterator :(
            (interval.0..(interval.1 + 1)).map(move |i| {
                if layer == 0 {
                    let v = self.config.convert_sample(data[i]);
                    (v, v)
                } else {
                    self.tree[self.layers[layer] + i]
                }
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static SMALL_DATA: [f32; 6] = [0., 1., -1., 0., 0.504, -0.504];
    fn make_small() -> SampleMipMap {
        assert_eq!(BASE, 4);
        let config = DisplayConfig {
            preamp: 1.,
            mode: DisplayMode::Linear,
        };
        SampleMipMap::new(&SMALL_DATA[..], config)
    }

    #[test]
    fn small_tree_construction() {
        let s = make_small();
        assert_eq!(s.size, 6);
        assert_eq!(s.tree, [(-127, 127), (-64, 64)]);
    }

    const EPS: f64 = 1e-4;
    #[test]
    fn small_query() {
        let s = make_small();
        let (a, b) = s.query(&SMALL_DATA[..], 2., 3., 3);
        let v: Vec<_> = b.collect();

        assert!(a.0 - (-0.5) <= EPS);
        assert!(a.1 - 3.5 <= EPS);
        assert_eq!(v, [(127, 127), (-127, -127), (0, 0), (64, 64), (-64, -64)]);

        let (a, b) = s.query(&SMALL_DATA[..], 0., 6., 1);
        let v: Vec<_> = b.collect();

        assert!(a.0 - 0.33333 <= EPS);
        assert!(a.1 - 1.0 <= EPS);
        assert_eq!(v, [(-127, 127), (-64, 64)]);
    }
}
