mod utils;

use std::ops::{Index, IndexMut};

use num_traits::{One, Zero};
use wasm_bindgen::prelude::*;

mod vect_s;

use vect_s::VectS;

// D => "dynamic", i.e. size known at run-time
// Wraps a std::Vec but gives us some convenience indexing.
#[derive(Debug, Clone)]
struct VectD<T> {
    data: Vec<T>,
}

impl<T> VectD<T> {
    fn zeros(size: usize) -> VectD<T>
    where
        T: Zero + Copy,
    {
        Self {
            data: vec![T::zero(); size],
        }
    }

    fn ones(size: usize) -> VectD<T>
    where
        T: One + Copy,
    {
        Self {
            data: vec![T::one(); size],
        }
    }

    fn map<S>(self, func: fn(T) -> S) -> VectD<S> {
        VectD {
            data: self.data.into_iter().map(func).collect(),
        }
    }
}

impl<T> Index<isize> for VectD<T> {
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        let index = if index >= 0 {
            index
        } else {
            self.data.len() as isize + index
        };

        return &self.data[index as usize];
    }
}

impl<T> IndexMut<isize> for VectD<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = if index >= 0 {
            index
        } else {
            self.data.len() as isize + index
        };

        return &mut self.data[index as usize];
    }
}

#[wasm_bindgen]
struct D1Q3 {
    nx: isize,

    f: VectD<VectS<f32, 3>>,

    rho: VectD<f32>,
    vel: VectD<f32>,
}

static Q: VectS<f32, 3> = VectS::new([-1.0, 0.0, 1.0]);
static W: VectS<f32, 3> = VectS::new([1. / 6., 4. / 6., 1. / 6.]);
const CS: f32 = 0.5773502691896258; // == 1/sqrt(3)

#[wasm_bindgen]
impl D1Q3 {
    pub fn new(nx: usize) -> D1Q3 {
        let mut out = D1Q3 {
            nx: nx as isize,
            f: VectD::zeros(nx),
            rho: VectD::ones(nx),
            vel: VectD::zeros(nx),
        };
        out.reinit();
        out
    }

    // Re-initialize the distribution function to the local equilibrium
    // as set by the macroscopic quantities.
    pub fn reinit(&mut self) {
        for i in 0..self.nx {
            self.f[i] = calc_f_eq(self.rho[i], self.vel[i], W, Q);
        }
    }

    pub fn stream(&mut self) {
        let tmp_l = self.f[0][0];
        let tmp_r = self.f[-1][2];
        for i in 0..self.nx - 1 {
            let j = i + 1;
            self.f[i][0] = self.f[j][0]; // left
        }
        // could easily be combined with loop above
        for i in (1..self.nx).rev() {
            let j = i - 1;
            self.f[i][2] = self.f[j][2]; // right
        }
        self.f[-1][0] = tmp_l;
        self.f[0][2] = tmp_r;
    }

    pub fn macro_(&mut self) {
        for i in 0..self.nx {
            self.rho[i] = self.f[i].sum();
            self.vel[i] = (self.f[i] * Q).sum() / (self.rho[i]);
        }
    }

    pub fn collide(&mut self, tau: f32) {
        for i in 0..self.nx {
            let f_eq = calc_f_eq(self.rho[i], self.vel[i], W, Q);
            self.f[i] = self.f[i] - (self.f[i] - f_eq) * (1.0 / tau);
        }
    }

    pub fn step(&mut self, tau: f32) {
        self.stream();
        self.macro_();
        self.collide(tau);
    }

    pub fn rho_(&mut self) -> *mut f32 {
        self.rho.data.as_mut_ptr()
    }

    pub fn vel_(&mut self) -> *mut f32 {
        self.vel.data.as_mut_ptr()
    }

    pub fn f_(&mut self) -> *mut VectS<f32, 3> {
        self.f.data.as_mut_ptr()
    }
}

fn calc_f_eq(rho: f32, vel: f32, w: VectS<f32, 3>, q: VectS<f32, 3>) -> VectS<f32, 3> {
    let mut out = VectS::zero();
    for d in 0..3 {
        out[d] = w[d]
            * rho
            * (1.0 + 3.0 * vel * q[d] - (3.0 / 2.0) * vel * vel + 4.5 * vel * vel * q[d] * q[d]);
    }
    return out;
}

fn sub_to_idx<const D: usize>(sub: VectS<usize, D>, counts: VectS<usize, D>) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in 0..D {
        idx += stride * sub[i];
        stride *= counts[i];
    }
    idx
}

// Split a (sufficiently small & positive) float into its integer and fractional parts.
fn split_int_frac(x: f32) -> (i32, f32) {
    let i = x.floor();
    let f = x - i;
    (i as i32, f)
}

struct Tracers<const D: usize> {
    pos: Vec<VectS<f32, D>>,
    vel: Vec<VectS<f32, D>>,
    extent: VectS<f32, D>,
    counts: VectS<usize, D>,
}

impl<const D: usize> Tracers<D> {
    pub fn new(size: usize, extent: VectS<f32, D>, counts: VectS<usize, D>) -> Self {
        return Self {
            pos: vec![VectS::zero(); size],
            vel: vec![VectS::zero(); size],
            extent: extent,
            counts: counts,
        };
    }

    /// Update the positions
    pub fn update(&mut self, dt: f32) {
        for i in 0..self.pos.len() {
            self.pos[i] = self.vel[i] * dt + self.pos[i];
        }
    }
}

#[wasm_bindgen]
struct Tracers1D {
    delegate: Tracers<1>,
}

impl Tracers1D {
    pub fn new(size: usize, extent: VectS<f32, 1>, counts: VectS<usize, 1>) -> Self {
        return Self {
            delegate: Tracers::<1>::new(size, extent, counts),
        };
    }

    /// TODO: At least the grid based initialization is easy to generalize to arbitrary D.
    pub fn reset(&mut self) {
        let dx = self.delegate.extent[0] / (self.delegate.pos.len() as f32);
        for i in 0..self.delegate.pos.len() {
            self.delegate.pos[i] = VectS::new([(i as f32 + 0.5) * dx]);
        }
    }

    // Linearly interpolate velocity for each tracer.
    pub fn interp_vel(&mut self, vel: Vec<VectS<f32, 1>>) {
        let dx = self.delegate.extent[0] / (self.delegate.counts[0] as f32);
        for i in 0..self.delegate.vel.len() {
            let frac = self.delegate.pos[i] / self.delegate.extent;
            let sub0 = (frac * (vel.len() as f32)).cast();
            let sub1 = VectS::new([sub0[0] + 1]);
            let idx0 = sub_to_idx(sub0, self.delegate.counts);
            let idx1 = sub_to_idx(sub1, self.delegate.counts);
            let frac = split_int_frac(self.delegate.pos[i][0] / dx).1;
            self.delegate.vel[i] = vel[idx0] * (1.0 - frac) + vel[idx1] * frac;
        }
    }
}

// shouldn't need to do this but meh

#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let cnt = VectS::new([30, 50, 70]);
        let sub = VectS::new([0, 0, 0]);
        assert_eq!(sub_to_idx(sub, cnt), 0);

        let sub = VectS::new([1, 0, 0]);
        assert_eq!(sub_to_idx(sub, cnt), 1);

        let sub = VectS::new([0, 1, 0]);
        assert_eq!(sub_to_idx(sub, cnt), 30);

        let sub = VectS::new([0, 0, 1]);
        assert_eq!(sub_to_idx(sub, cnt), 30 * 50);

        let sub = VectS::new([29, 49, 69]);
        assert_eq!(sub_to_idx(sub, cnt), 30 * 50 * 70 - 1);
    }
}
