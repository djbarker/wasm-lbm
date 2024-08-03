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

    f1: VectD<VectS<f32, 3>>,
    f2: VectD<VectS<f32, 3>>,

    rho: VectD<f32>,
    vel: VectD<f32>,

    t: f32,
}

static Q: VectS<f32, 3> = VectS::new([-1.0, 0.0, 1.0]);
static W: VectS<f32, 3> = VectS::new([1. / 6., 4. / 6., 1. / 6.]);
const CS: f32 = 0.5773502691896258; // == 1/sqrt(3)

#[wasm_bindgen]
impl D1Q3 {
    pub fn new(nx: usize, t: f32) -> D1Q3 {
        let mut out = D1Q3 {
            nx: nx as isize,
            f1: VectD::zeros(nx),
            f2: VectD::zeros(nx),
            rho: VectD::ones(nx),
            vel: VectD::zeros(nx),
            t: t * CS * CS,
        };
        out.reinit();
        out
    }

    // Re-initialize the distribution function to the local equilibrium
    // as set by the macroscopic quantities.
    pub fn reinit(&mut self) {
        for i in 0..self.nx {
            self.f1[i] = calc_f_eq(self.rho[i], self.vel[i], self.t, W, Q);
        }
    }

    pub fn stream(&mut self) {
        let tmp_l = self.f1[0][0];
        let tmp_r = self.f1[-1][2];
        for i in 0..self.nx - 1 {
            let j = i + 1;
            self.f1[i][0] = self.f1[j][0]; // left
        }
        // could easily be combined with loop above
        for i in (1..self.nx).rev() {
            let j = i - 1;
            self.f1[i][2] = self.f1[j][2]; // right
        }
        self.f1[-1][0] = tmp_l;
        self.f1[0][2] = tmp_r;
    }

    pub fn macro_(&mut self) {
        for i in 0..self.nx {
            self.rho[i] = self.f1[i].sum();
            self.vel[i] = (self.f1[i] * Q).sum() / (self.rho[i]);
        }
    }

    pub fn collide(&mut self, tau: f32) {
        for i in 0..self.nx {
            let f_eq = calc_f_eq(self.rho[i], self.vel[i], self.t, W, Q);
            self.f1[i] = self.f1[i] - (self.f1[i] - f_eq) * (1.0 / tau);
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
        self.f1.data.as_mut_ptr()
    }
}

fn calc_f_eq(rho: f32, vel: f32, t: f32, w: VectS<f32, 3>, q: VectS<f32, 3>) -> VectS<f32, 3> {
    let mut out = VectS::zero();
    for d in 0..3 {
        // rho * (2/3) * (1 + (1/2) * u^2 / c^2) * (-1));
        // rho * (1/3) * (2 - u^2 / c^2);
        out[d] = w[d]
            * rho
            * (1.0 + 3.0 * vel * q[d] - (3.0 / 2.0) * vel * vel + 4.5 * vel * vel * q[d] * q[d]);
    }
    return out;
}

// shouldn't need to do this but meh

#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}
