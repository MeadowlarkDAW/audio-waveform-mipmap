use std::cmp::{max, min};
use std::vec::Vec;

pub struct WaveformConfig {
    preamp: f32,
    mode: WaveformMode,
}

pub enum WaveformMode {
    Linear,
    DB(f32),
}

impl Default for WaveformConfig {
    fn default() -> Self {
        WaveformConfig {
            preamp: 1.0,
            mode: WaveformMode::Linear,
        }
    }
}

impl WaveformConfig {
    #[inline]
    // Generic over value type?
    pub fn convert_sample(&self, value: f32) -> f32 {
        let value = value * self.preamp;
        match self.mode {
            WaveformMode::Linear => value.max(-1.).min(1.),
            WaveformMode::DB(lowest) => {
                let db_value = 20. * value.abs().log10();
                let scaled = ((db_value / lowest) + 1.).max(0.).min(1.);
                scaled.copysign(value)
            }
        }
    }
    #[inline]
    pub fn compress_sample(&self, value: f32) -> i8 {
        (self.convert_sample(value) * std::i8::MAX as f32) as i8
    }
}

pub const BASE: usize = 4;
pub struct Waveform {
    size: usize,
    config: WaveformConfig,
    tree: Vec<(i8, i8)>,
    layers: Vec<usize>,
}

impl Default for Waveform {
    fn default() -> Self {
        Waveform {
            size: 0,
            config: Default::default(),
            tree: Vec::new(),
            layers: Vec::new(),
        }
    }
}

impl Waveform {
    pub fn new(data: &[f32], config: WaveformConfig) -> Self {
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
                            .map(|v| config.compress_sample(*v))
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
                                (min(a.0, b.0), max(a.1, b.1))
                            }),
                    );
                }
            }
            layers.push(pos);
            pos += current_layer;
        }
        Waveform {
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
    /// Returns an interator yielding `(x, min, max)`. `x` is in pixel space.
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
    ) -> impl ExactSizeIterator<Item = (f32, f32, f32)> + 'a {
        assert!(data.len() == self.size);
        
        let padding = width / (pixels as f64);
        let mut interval = (
            (((left - padding).ceil() as i64).max(0) as usize).min(self.size - 1),
            (((left + width + padding).floor() as i64).max(0) as usize).min(self.size - 1),
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
            ((x - left) / width * (pixels as f64)) as f32
        };
        let l_px = layer_index_to_px(interval.0);
        let w_px_per_step = (layer_index_to_px(interval.1) - l_px) / (interval.1 - interval.0) as f32;

        let offset = self.layers[layer];
        // RangeInclusive<usize>: !ExactSizeIterator :(
        (interval.0..(interval.1 + 1)).map(move |i| {
            let x_px = l_px + w_px_per_step * (i - interval.0) as f32;
            if layer == 0 {
                let v = self.config.convert_sample(data[i]);
                (x_px, v, v)
            } else {
                let p = self.tree[offset + i];
                (x_px, p.0 as f32 / std::i8::MAX as f32, p.1 as f32 / std::i8::MAX as f32)
            }
        })
    }
    
    /// Query a range of data exactly.
    ///
    /// Produces exactly `pixels + 2` points unless querying out of bounds.
    pub fn query_exact<'a>(&'a self, data: &'a [f32], left: f64, width: f64, pixels: usize)
        -> impl ExactSizeIterator<Item = (f32, f32, f32)> + 'a {
        assert!(data.len() == self.size);
        
        let samples_per_px = width / pixels as f64;
        let max_px = (self.size as f64 / samples_per_px) as i64;
        let left_px = (left / samples_per_px) as i64;
        let interval = (
            (left_px - 1).max(0).min(max_px),
            (left_px + pixels as i64 + 1).max(0).min(max_px),
        );
        (interval.0 as usize..interval.1 as usize).map(move |i| {
            let x_px = (i as i64 - left_px) as f32 + 0.5;
            let p = self.range_min_max(data, (i as f64 * samples_per_px) as usize, ((i+1) as f64 * samples_per_px) as usize);
            (x_px, p.0 as f32 / std::i8::MAX as f32, p.1 as f32 / std::i8::MAX as f32)
        })
    }
    
    /// Returns minimum and maximum in the sample range [left, right).
    ///
    /// O(BASE * log_BASE(right - left))
    fn range_min_max(&self, data: &[f32], mut left: usize, mut right: usize) -> (i8, i8) {
        left = max(left, 0);
        right = min(right, self.size);
        
        let mut result = (std::i8::MAX, std::i8::MIN);
        while left < right && left % BASE != 0 {
            let v = self.config.compress_sample(data[left]);
            result = (min(result.0, v), max(result.1, v));
            left += 1;
        }
        while left < right && right % BASE != 0 {
            right -= 1;
            let v = self.config.compress_sample(data[right]);
            result = (min(result.0, v), max(result.1, v));
        }
        let mut layer_iter = self.layers.iter().skip(1);
        while left < right {
            left /= BASE;
            right /= BASE;
            let layer = &self.tree[*layer_iter.next().unwrap()..];
            while left < right && left % BASE != 0 {
                let p = layer[left];
                result = (min(result.0, p.0), max(result.1, p.1));
                left += 1;
            }
            while left < right && right % BASE != 0 {
                right -= 1;
                let p = layer[right];
                result = (min(result.0, p.0), max(result.1, p.1));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    static SMALL_DATA: [f32; 6] = [0., 1., -1., 0., 0.504, -0.504];
    fn make_small() -> Waveform {
        assert_eq!(BASE, 4);
        let config = WaveformConfig {
            preamp: 1.,
            mode: WaveformMode::Linear,
        };
        Waveform::new(&SMALL_DATA[..], config)
    }

    #[test]
    fn small_tree_construction() {
        let s = make_small();
        assert_eq!(s.size, 6);
        assert_eq!(s.tree, [(-127, 127), (-64, 64)]);
    }

    const EPS: f32 = 1e-2;
    fn assert_f32_eq(a: f32, b: f32) {
        if (a-b).abs() > EPS {
            panic!("f64 equality assertion failed: {} != {}", a, b);
        }
    }
    fn all_eq(a: impl ExactSizeIterator<Item = (f32, f32, f32)>, b: &[(f32, f32, f32)]) {
        assert_eq!(a.len(), b.len());
        for (u, v) in a.zip(b) {
            assert_f32_eq(u.0, v.0);
            assert_f32_eq(u.1, v.1);
            assert_f32_eq(u.2, v.2);
        }
    }
    #[test]
    fn small_query() {
        let s = make_small();
        
        let it = s.query(&SMALL_DATA[..], 2., 3., 3);
        all_eq(it, &[(-0.5, 1.0, 1.0), (0.5, -1.0, -1.0), (1.5, 0.0, 0.0), (2.5, 0.504, 0.504), (3.5, -0.504, -0.504)]);

        let it = s.query(&SMALL_DATA[..], 0., 6., 1);
        all_eq(it, &[(0.3333, -1.0, 1.0), (1.0, -0.5, 0.5)]);
    }
    
    #[test]
    fn small_query_exact() {
        let s = make_small();
        
        let it = s.query_exact(&SMALL_DATA[..], 2., 3., 3);
        all_eq(it, &[(-0.5, 1.0, 1.0), (0.5, -1.0, -1.0), (1.5, 0.0, 0.0), (2.5, 0.504, 0.504), (3.5, -0.504, -0.504)]);
        
        let it = s.query_exact(&SMALL_DATA[..], 0., 6., 2);
        all_eq(it, &[(0.5, -1.0, 1.0), (1.5, -0.504, 0.504)]);
    }
}
