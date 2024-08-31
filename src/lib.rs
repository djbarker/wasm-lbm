mod utils;

use std::ops::{Add, Rem};

use num_traits::{One, Zero};
use wasm_bindgen::prelude::*;

mod vect_d;
mod vect_s;

use vect_d::VectD;
use vect_s::VectS;

fn sub_to_idx<const D: usize>(sub: VectS<isize, D>, counts: VectS<isize, D>) -> isize {
    let mut idx = 0;
    let mut stride = 1;
    for i in 0..D {
        idx += stride * sub[i];
        stride *= counts[i] as isize;
    }
    idx
}

fn raster<const D: usize>(mut sub: VectS<isize, D>, counts: VectS<isize, D>) -> VectS<isize, D> {
    sub[0] += 1;
    for d in 0..(D - 1) {
        if sub[d] == counts[d] {
            sub[d] = 0;
            sub[d + 1] += 1;
        }
    }
    // if sub[D - 1] == counts[D - 1] {
    //     panic!("Raster past end!")
    // }
    sub
}

fn raster_end<const D: usize>(counts: VectS<isize, D>) -> VectS<isize, D> {
    let end = counts - VectS::one();
    raster(end, counts)
}

// Split a (sufficiently small & positive) float into its integer and fractional parts.
fn split_int_frac(x: f32) -> (i32, f32) {
    let i = x.floor();
    let f = x - i;
    (i as i32, f)
}

/// Modulo operation which handles -ve `x`.
fn fmod<T>(x: T, m: T) -> T
where
    T: Rem<Output = T> + Add<Output = T> + Copy,
{
    (x + m) % m
}

/// Elementwise modulo the components of `x` with those of `m`.  
fn vmod<const D: usize, T>(x: VectS<T, D>, m: VectS<T, D>) -> VectS<T, D>
where
    T: Rem<Output = T> + Add<Output = T> + Copy + Default + 'static,
{
    x.map_with_idx(|j, x| fmod(x, m[j]))
}

/// Calculate the approximate equilibrium distribution for the given density,
/// velocity & lattice weight/velocity set.
fn calc_f_eq<const D: usize, const Q: usize>(
    rho: f32,
    vel: VectS<f32, D>,
    ws: [f32; Q],
    qs: [VectS<f32, D>; Q],
) -> VectS<f32, Q> {
    let vv = (vel * vel).sum();
    let mut out = VectS::zero();
    for i in 0..Q {
        let vq = (vel * qs[i]).sum();
        out[i] = ws[i] * rho * (1.0 + 3.0 * vq - (3.0 / 2.0) * vv + 4.5 * vq * vq);
    }
    return out;
}

struct LBM<const D: usize, const Q: usize> {
    ws: [f32; Q],
    qs: [VectS<f32, D>; Q],

    cnt: VectS<isize, D>,

    f1: VectD<VectS<f32, Q>>,
    f2: VectD<VectS<f32, Q>>,

    rho: VectD<f32>,
    vel: VectD<VectS<f32, D>>,
}

impl<const D: usize, const Q: usize> LBM<D, Q> {
    pub fn new(cnt: VectS<usize, D>, ws: [f32; Q], qs: [VectS<f32, D>; Q]) -> LBM<D, Q> {
        let n = cnt.prod();
        let mut out = LBM {
            ws: ws,
            qs: qs,
            cnt: cnt.cast(),
            f1: VectD::zeros(n),
            f2: VectD::zeros(n),
            rho: VectD::ones(n),
            vel: VectD::zeros(n),
        };
        out.reinit();
        out
    }

    // Re-initialize the distribution function to the local equilibrium
    // as set by the macroscopic quantities.
    pub fn reinit(&mut self) {
        for i in 0..self.cnt.prod() {
            self.f1[i] = calc_f_eq(self.rho[i], self.vel[i], self.ws, self.qs);
        }
    }

    pub fn step(&mut self, tau: f32) {
        let mut sub = VectS::zero();
        for i in 0..self.cnt.prod() {
            // stream
            for q in 0..Q {
                let sub_ = sub - self.qs[q].cast();
                let sub_ = vmod(sub_, self.cnt);
                let j = sub_to_idx(sub_, self.cnt);
                self.f2[i][q] = self.f1[j][q];
            }

            // macro
            self.rho[i] = self.f2[i].sum();
            self.vel[i] = self.f2[i].map_with_idx(|i, f| f * self.qs[i]).sum() / self.rho[i];

            // collide
            let f_eq = calc_f_eq(self.rho[i], self.vel[i], self.ws, self.qs);
            self.f2[i] = self.f2[i] - (self.f2[i] - f_eq) * (1.0 / tau);

            sub = raster(sub, self.cnt);
        }

        std::mem::swap(&mut self.f1, &mut self.f2);
    }
}

#[wasm_bindgen]
struct D1Q3 {
    lbm: LBM<1, 3>,
}

static D1Q3_Q: [VectS<f32, 1>; 3] = [VectS::new([-1.0]), VectS::new([0.0]), VectS::new([1.0])];
static D1Q3_W: [f32; 3] = [1. / 6., 4. / 6., 1. / 6.];

#[wasm_bindgen]
impl D1Q3 {
    pub fn new(nx: usize) -> D1Q3 {
        D1Q3 {
            lbm: LBM::new(VectS::new([nx]), D1Q3_W, D1Q3_Q),
        }
    }

    // Re-initialize the distribution function to the local equilibrium
    // as set by the macroscopic quantities.
    pub fn reinit(&mut self) {
        self.lbm.reinit()
    }

    pub fn step_n(&mut self, tau: f32, n: usize) {
        for _ in 0..n {
            self.step(tau);
        }
    }

    pub fn step(&mut self, tau: f32) {
        self.lbm.step(tau)
    }

    pub fn rho_(&mut self) -> *mut f32 {
        self.lbm.rho.data.as_mut_ptr()
    }

    pub fn vel_(&mut self) -> *mut f32 {
        self.lbm.vel.data.as_mut_ptr() as *mut f32
    }
}

/// Take the tensor product of two velocity sets.
///
/// #### NOTE
///
/// In Rust we cannot use expressions of const generics as const generic args themselves.
/// Thus it is necessary to have the output D & Q explicitly as arguments.
/// However, we usually don't need to specify the generic arguments because the type inference
/// works it out for us.
/// Annoyingly, if your upstream use has the wrong D & Q values this will compile,
/// but you will at least get a runtime assertion error.
fn tensor_prod_q<
    const D1: usize,
    const Q1: usize,
    const D2: usize,
    const Q2: usize,
    const D3: usize,
    const Q3: usize,
>(
    q1s: [VectS<f32, D1>; Q1],
    q2s: [VectS<f32, D2>; Q2],
) -> [VectS<f32, D3>; Q3] {
    // can't use const expressions as generic const args
    assert_eq!(D1 + D2, D3);
    assert_eq!(Q1 * Q2, Q3);
    let mut out = [VectS::zero(); Q3];
    let mut i = 0;
    for q1 in 0..Q1 {
        for q2 in 0..Q2 {
            for d in 0..D1 {
                out[i][d] = q1s[q1][d];
            }
            for d in 0..D2 {
                out[i][D1 + d] = q2s[q2][d];
            }
            i += 1;
        }
    }
    out
}

/// Take the tensor product of two weight sets.
///
/// #### NOTE
///
/// See the note on `tensor_prod_q` about generic const arguments.
fn tensor_prod_w<const Q1: usize, const Q2: usize, const Q3: usize>(
    w1s: [f32; Q1],
    w2s: [f32; Q2],
) -> [f32; Q3] {
    // can't use const expressions as generic const args
    assert_eq!(Q1 * Q2, Q3);
    let mut out = [0.0; Q3];
    let mut sum = 0.0;
    let mut i = 0;
    for q1 in 0..Q1 {
        for q2 in 0..Q2 {
            out[i] = w1s[q1] * w2s[q2];
            sum += out[i];
            i += 1;
        }
    }
    // renormalize
    for i in 0..Q3 {
        out[i] /= sum;
    }
    out
}

#[wasm_bindgen]
struct D2Q9 {
    lbm: LBM<2, 9>,
}

#[wasm_bindgen]
impl D2Q9 {
    pub fn new(nx: usize, ny: usize) -> D2Q9 {
        // Unfortunately we cannot call tensor_prod_* in a static context.
        // So we cannot have D2Q9_* static variables a la D1Q3_*.
        let d2q9_q = tensor_prod_q(D1Q3_Q, D1Q3_Q);
        let d2q9_w = tensor_prod_w(D1Q3_W, D1Q3_W);

        D2Q9 {
            lbm: LBM::new(VectS::new([nx, ny]), d2q9_w, d2q9_q),
        }
    }

    // Re-initialize the distribution function to the local equilibrium
    // as set by the macroscopic quantities.
    pub fn reinit(&mut self) {
        self.lbm.reinit()
    }

    pub fn step_n(&mut self, tau: f32, n: usize) {
        for _ in 0..n {
            self.step(tau);
        }
    }

    pub fn step(&mut self, tau: f32) {
        self.lbm.step(tau)
    }

    pub fn rho_(&mut self) -> *mut f32 {
        self.lbm.rho.data.as_mut_ptr()
    }

    pub fn vel_(&mut self) -> *mut f32 {
        self.lbm.vel.data.as_mut_ptr() as *mut f32
    }
}

struct Tracers<const D: usize> {
    pos: VectD<VectS<f32, D>>,
    vel: VectD<VectS<f32, D>>,
    extent: VectS<f32, D>,
    counts: VectS<isize, D>,
}

impl<const D: usize> Tracers<D> {
    pub fn new(size: usize, extent: VectS<f32, D>, counts: VectS<isize, D>) -> Self {
        return Self {
            pos: VectD::zeros(size),
            vel: VectD::zeros(size),
            extent: extent,
            counts: counts,
        };
    }

    /// Update the positions
    pub fn update(&mut self, dt: f32) {
        for i in 0..self.pos.len() {
            self.pos[i] = self.pos[i] + self.vel[i] * dt;
            self.pos[i] = vmod(self.pos[i], self.extent);
            // for j in 0..D {
            //     self.pos[i][j] = (self.pos[i][j] + self.extent[j]) % self.extent[j];
            // }
        }
    }
}

#[wasm_bindgen]
struct Tracers1D {
    delegate: Tracers<1>,
}

#[wasm_bindgen]
impl Tracers1D {
    pub fn new(size: usize, extent: f32, count: isize) -> Self {
        return Self {
            delegate: Tracers::<1>::new(size, VectS::new([extent]), VectS::new([count])),
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
    pub fn interp_vel(&mut self, lbm: &D1Q3) {
        let dx = self.delegate.extent[0] / (self.delegate.counts[0] as f32);
        for i in 0..self.delegate.vel.len() {
            let frac = self.delegate.pos[i] / self.delegate.extent;
            let sub0 = (frac * (lbm.lbm.cnt.prod() as f32)).cast::<isize>();
            let sub1 = sub0 + VectS::new([1]);
            let sub1 = vmod(sub1, self.delegate.counts.cast());
            let idx0 = sub_to_idx(sub0, self.delegate.counts);
            let idx1 = sub_to_idx(sub1, self.delegate.counts);
            let frac = split_int_frac(self.delegate.pos[i][0] / dx).1;
            self.delegate.vel[i] = lbm.lbm.vel[idx0] * (1.0 - frac) + lbm.lbm.vel[idx1] * frac;
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.delegate.update(dt)
    }

    /// Expose the 1D position array to JS
    pub fn pos_(&mut self) -> *mut f32 {
        self.delegate.pos.data.as_mut_ptr() as *mut f32
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
    fn test_sub_to_idx() {
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

    #[test]
    fn test_raster() {
        let cnt = VectS::new([2, 2, 2]);
        let mut sub = VectS::new([0, 0, 0]);

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 0, 0]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([0, 1, 0]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 1, 0]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([0, 0, 1]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 0, 1]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([0, 1, 1]));

        sub = raster(sub, cnt);
        assert_eq!(sub, VectS::new([1, 1, 1]));

        // TODO: check one more panics
        // sub = raster(sub, cnt);
    }
}
