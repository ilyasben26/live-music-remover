//! Experimental noise injection, shared by all pipelines:
//! - input noise: pink noise added before the model, in an effort to reduce
//!   musical artefacts;
//! - comfort noise: pink noise added after the model, to mask musical artefacts.
//!
//! Levels are in dB relative to the input level (a power average over the last
//! few seconds), so silence gets no noise. `OFF_DB` or below disables either one.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Slider value that means "off".
pub const OFF_DB: f32 = -80.0;
/// Time constant of the input level tracker, in seconds.
const LEVEL_TAU_S: f32 = 3.0;

/// Noise levels set from the UI, in dB relative to the input level.
#[derive(Clone)]
pub struct NoiseControls {
    pub input_db: Arc<AtomicU32>,
    pub comfort_db: Arc<AtomicU32>,
}

impl Default for NoiseControls {
    fn default() -> Self {
        Self {
            input_db: Arc::new(AtomicU32::new(OFF_DB.to_bits())),
            comfort_db: Arc::new(AtomicU32::new(OFF_DB.to_bits())),
        }
    }
}

fn load_db(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

/// Pink noise with unit RMS (Paul Kellet's filter on white noise).
struct PinkNoise {
    rng: u64,
    b: [f32; 7],
    scale: f32,
}

impl PinkNoise {
    fn new(seed: u64) -> Self {
        let mut p = Self {
            rng: seed | 1,
            b: [0.0; 7],
            scale: 1.0,
        };
        // Calibrate to unit RMS.
        let n = 1 << 16;
        let power: f32 = (0..n).map(|_| p.next().powi(2)).sum::<f32>() / n as f32;
        p.scale = 1.0 / power.sqrt();
        p
    }

    /// Uniform white noise in [-1, 1) (xorshift64).
    fn white(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn next(&mut self) -> f32 {
        let w = self.white();
        let b = &mut self.b;
        b[0] = 0.99886 * b[0] + w * 0.0555179;
        b[1] = 0.99332 * b[1] + w * 0.0750759;
        b[2] = 0.96900 * b[2] + w * 0.1538520;
        b[3] = 0.86650 * b[3] + w * 0.3104856;
        b[4] = 0.55000 * b[4] + w * 0.5329522;
        b[5] = -0.7616 * b[5] - w * 0.0168980;
        let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
        b[6] = w * 0.115926;
        pink * self.scale
    }
}

pub struct NoiseInjector {
    controls: NoiseControls,
    input: PinkNoise,
    comfort: PinkNoise,
    /// Tracked input power (mean square).
    level_pow: f32,
    /// Samples per second across all channels, to time the level tracker.
    samples_per_s: f32,
}

impl NoiseInjector {
    pub fn new(controls: NoiseControls, sample_rate: usize, channels: usize) -> Self {
        Self {
            controls,
            input: PinkNoise::new(0x9E37_79B9_7F4A_7C15),
            comfort: PinkNoise::new(0xD1B5_4A32_D192_ED03),
            level_pow: 0.0,
            samples_per_s: (sample_rate * channels) as f32,
        }
    }

    /// Tracks the level of the (clean) input, then adds input noise to it.
    /// Call once per block, before the model.
    pub fn process_input(&mut self, x: &mut [f32]) {
        if x.is_empty() {
            return;
        }
        let pow = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let a = (-(x.len() as f32) / (self.samples_per_s * LEVEL_TAU_S)).exp();
        self.level_pow = a * self.level_pow + (1.0 - a) * pow;
        let db = load_db(&self.controls.input_db);
        if db > OFF_DB {
            let gain = self.level_pow.sqrt() * 10f32.powf(db / 20.0);
            for v in x.iter_mut() {
                *v += self.input.next() * gain;
            }
        }
    }

    /// Adds comfort noise to the model output. Call once per block, after the model.
    pub fn process_output(&mut self, y: &mut [f32]) {
        let db = load_db(&self.controls.comfort_db);
        if db > OFF_DB {
            let gain = self.level_pow.sqrt() * 10f32.powf(db / 20.0);
            for v in y.iter_mut() {
                *v += self.comfort.next() * gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_the_input_and_off_adds_nothing() {
        let controls = NoiseControls::default();
        let mut n = NoiseInjector::new(controls.clone(), 48_000, 1);
        let tone = |i: usize| 0.1 * (i as f32 * 0.07).sin(); // RMS ~0.0707
        let mut out = vec![0.0f32; 480];
        // Off: untouched.
        for k in 0..1000 {
            let mut x: Vec<f32> = (0..480).map(|i| tone(k * 480 + i)).collect();
            let orig = x.clone();
            n.process_input(&mut x);
            n.process_output(&mut out);
            assert_eq!(x, orig);
        }
        assert!(out.iter().all(|&v| v == 0.0));
        // Comfort noise at -20 dB: RMS ~0.00707 once the level has settled.
        controls.comfort_db.store((-20.0f32).to_bits(), Ordering::Relaxed);
        let mut power = 0.0;
        for k in 0..200 {
            let mut x: Vec<f32> = (0..480).map(|i| tone(k * 480 + i)).collect();
            n.process_input(&mut x);
            out.fill(0.0);
            n.process_output(&mut out);
            power += out.iter().map(|v| v * v).sum::<f32>();
        }
        let rms = (power / (200.0 * 480.0)).sqrt();
        assert!((rms / 0.00707 - 1.0).abs() < 0.15, "rms {rms}");
    }
}
