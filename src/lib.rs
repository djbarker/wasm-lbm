mod raster;
mod utils;

use num_traits::{One, Zero};
use raster::{sub_to_idx, Raster};
use utils::{fmod, split_int_frac, vmod};
use wasm_bindgen::prelude::*;

mod vect_d;
mod vect_s;

use vect_d::VectD;
use vect_s::VectS;

// re-exports
pub use wasm_bindgen::memory;
// pub use wasm_bindgen_rayon::init_thread_pool;

fn copy_periodic<T, const D: usize>(arr: &mut VectD<T>, cnt_pad: VectS<isize, D>)
where
    T: Copy,
{
    if D == 1 {
        // xlower -> xupper:
        arr[cnt_pad[0] - 1] = arr[1];
        // xupper -> xlower:
        arr[0] = arr[cnt_pad[0] - 2];
    } else if D == 2 {
        for yidx in 0..cnt_pad[1] {
            // xlower -> xupper:
            arr[yidx * cnt_pad[0] + (cnt_pad[0] - 1)] = arr[yidx * cnt_pad[0] + 1];
            // xupper -> xlower:
            arr[yidx * cnt_pad[0] + 0] = arr[yidx * cnt_pad[0] + (cnt_pad[0] - 2)];
        }
        for xidx in 0..cnt_pad[0] {
            // ylower -> yupper:
            arr[(cnt_pad[1] - 1) * cnt_pad[0] + xidx] = arr[1 * cnt_pad[0] + xidx];
            // yupper -> ylower:
            arr[0 * cnt_pad[0] + xidx] = arr[(cnt_pad[1] - 2) * cnt_pad[0] + xidx];
        }
    } else {
        // TODO: Would be nice to make this generic, or at least have the 3D code handle 2D and 1D
        //       by faking width (& depth) of 1 cell.
        panic!()
    }
}

/// Calculate the approximate equilibrium distribution for the given density,
/// velocity & lattice weight/velocity set.
#[rustfmt::skip]
fn calc_f_eq<const D: usize, const Q: usize>(
    rho: f32,
    vel: VectS<f32, D>,
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
) -> VectS<f32, Q> {
    let vv = (vel * vel).sum();
    
    // if (D == 2) && (Q == 9) && false {
    //     // Explicitly write out D2Q9 feq calculation
        
    //     let v = vel;
    //     let mut out = VectS::zero();
    
    //     let vxx = v[0] * v[0];
    //     let vyy = v[1] * v[1];
    //     let vxy = v[0] * v[1];

    //     // 0:        1:       2:       3:       4:      5:      6:       7:      8:
    //     // [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]
    //     out[4] = rho * (2.0 / 9.0)  * (2.0 - 3.0 * vv);
    //     out[7] = rho * (1.0 / 18.0) * (2.0 + 6.0 * v[0] + 9.0 * vxx - 3.0 * vv);
    //     out[1] = rho * (1.0 / 18.0) * (2.0 - 6.0 * v[0] + 9.0 * vxx - 3.0 * vv);
    //     out[5] = rho * (1.0 / 18.0) * (2.0 + 6.0 * v[1] + 9.0 * vyy - 3.0 * vv);
    //     out[3] = rho * (1.0 / 18.0) * (2.0 - 6.0 * v[1] + 9.0 * vyy - 3.0 * vv);
    //     out[8] = rho * (1.0 / 36.0) * (1.0 + 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv);
    //     out[0] = rho * (1.0 / 36.0) * (1.0 - 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv);
    //     out[2] = rho * (1.0 / 36.0) * (1.0 + 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv);
    //     out[6] = rho * (1.0 / 36.0) * (1.0 - 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv);
        
    //     out
    // } else {
        let mut out = ws;
        for i in 0..Q {
            let vq = (vel * qs[i]).sum();
            out[i] *= rho * (1.0 + 3.0 * vq - 1.5 * vv + 4.5 * vq * vq);
        }
        out
    // }
}

fn update_generic_bgk<const D: usize, const Q: usize>(
    even: bool,
    omega: f32,
    f: &mut VectD<VectS<f32, Q>>,
    rho: &mut VectD<f32>,
    vel: &mut VectD<VectS<f32, D>>,
    idx: VectS<isize, Q>,
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
    js: [usize; Q],
) {
    // collect fs
    let mut f_: VectS<f32, Q> = VectS::default();
    for i in 0..Q {
        f_[i] = if even { f[idx[i]][i] } else { f[idx[0]][js[i]] };
    }

    // calc moments
    let r = f_.sum();
    let mut v = VectS::<f32, D>::zero();
    for i in 0..Q {
        v += f_[i] * qs[i] / r;
    }
    let v = v; // no mut
    let vv = (v * v).sum();

    // calc equilibrium & collide
    for i in 0..Q {
        let vq = (v * qs[i]).sum();
        let feq = r * ws[i] * (1.0 + 3.0 * vq - 1.5 * vv + 4.5 * vq * vq);
        f_[i] += omega * (feq - f_[i]);
    }
    // let feq = calc_f_eq(r, v, ws, qs);  // TODO: calc_f_eq uses different index convention for D2Q9
    // f_ += omega * (feq - f_);

    // write back to same locations
    for i in 0..Q {
        if even {
            let j = js[i];
            f[idx[j]][j] = f_[i];
        } else {
            f[idx[0]][i] = f_[i];
        }
    }

    // update the macroscopic observables
    rho[idx[0]] = r;
    vel[idx[0]] = v;
}

fn update_d1q3_bgk<const D: usize, const Q: usize>(
    even: bool,
    omega: f32,
    f: &mut VectD<VectS<f32, Q>>,
    rho: &mut VectD<f32>,
    vel: &mut VectD<VectS<f32, D>>,
    idx: VectS<isize, Q>,
) {
    // Boo; we have to make this function generic but only want D1Q3.
    assert_eq!(D, 1);
    assert_eq!(Q, 3);

    // collect fs
    let (f0, f1, f2) = if even {
        (f[idx[0]][0], f[idx[1]][1], f[idx[2]][2])
    } else {
        (f[idx[0]][0], f[idx[0]][2], f[idx[0]][1])
    };

    // calc moments
    let r = f0 + f1 + f2;
    let v = (f1 - f2) / r;
    let vv = v * v;

    // calc equilibrium
    let f0eq = r * (1. / 3.) * (2. - 3. * vv);
    let f1eq = r * (1. / 12.) * (2. + 6. * v + 6. * vv);
    let f2eq = r * (1. / 12.) * (2. - 6. * v + 6. * vv);

    // write back to same locations
    if even {
        f[idx[0]][0] = f0 + omega * (f0eq - f0);
        f[idx[2]][2] = f1 + omega * (f1eq - f1);
        f[idx[1]][1] = f2 + omega * (f2eq - f2);
    } else {
        f[idx[0]][0] = f0 + omega * (f0eq - f0);
        f[idx[0]][1] = f1 + omega * (f1eq - f1);
        f[idx[0]][2] = f2 + omega * (f2eq - f2);
    }

    rho[idx[0]] = r;
    vel[idx[0]][0] = v;
}

#[rustfmt::skip]
fn update_d2q9_bgk<const D: usize, const Q: usize>(
    even: bool,
    omega: f32,
    f: &mut VectD<VectS<f32, Q>>,
    rho: &mut VectD<f32>,
    vel: &mut VectD<VectS<f32, D>>,
    idx: VectS<isize, Q>,
) {
    // Boo; we have to make this function generic but only want D2Q9.
    assert_eq!(D, 2);
    assert_eq!(Q, 9);

    // collect fs
    let mut f_ = VectS::new(if even {
        [
            f[idx[0]][0],
            f[idx[1]][1],
            f[idx[2]][2],
            f[idx[3]][3],
            f[idx[4]][4],
            f[idx[5]][5],
            f[idx[6]][6],
            f[idx[7]][7],
            f[idx[8]][8],
        ]
    } else {
        [
            f[idx[0]][0],
            f[idx[0]][2],
            f[idx[0]][1],
            f[idx[0]][6],
            f[idx[0]][8],
            f[idx[0]][7],
            f[idx[0]][3],
            f[idx[0]][5],
            f[idx[0]][4],
        ]
    });

    // 0:  0  0
    // 1:  0 +1
    // 2:  0 -1
    // 3: +1  0
    // 4: +1 +1
    // 5: +1 -1
    // 6: -1  0
    // 7: -1 +1
    // 8: -1 -1

    // calc moments
    let r = f_.sum();
    let mut v: VectS<f32, D> = VectS::zero();
    v[0] = (f_[3] + f_[4] + f_[5] - f_[6] - f_[7] - f_[8]) / r;
    v[1] = (f_[1] - f_[2] + f_[4] - f_[5] + f_[7] - f_[8]) / r;
    let vv = (v * v).sum();
    let vxx = v[0] * v[0];
    let vyy = v[1] * v[1];
    let vxy = v[0] * v[1];

    // calc equilibrium & collide
    f_[0] += omega * (r * (2.0 / 9.0) * (2.0 - 3.0 * vv) - f_[0]);
    f_[1] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[1]);
    f_[2] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * v[1] + 9.0 * vyy - 3.0 * vv) - f_[2]);
    f_[3] += omega * (r * (1.0 / 18.0) * (2.0 + 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[3]);
    f_[4] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv) - f_[4]);
    f_[5] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv) - f_[5]);
    f_[6] += omega * (r * (1.0 / 18.0) * (2.0 - 6.0 * v[0] + 9.0 * vxx - 3.0 * vv) - f_[6]);
    f_[7] += omega * (r * (1.0 / 36.0) * (1.0 + 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv) - f_[7]);
    f_[8] += omega * (r * (1.0 / 36.0) * (1.0 - 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv) - f_[8]);

    // write back to same locations
    if even {
        f[idx[0]][0] = f_[0];
        f[idx[2]][2] = f_[1];
        f[idx[1]][1] = f_[2];
        f[idx[6]][6] = f_[3];
        f[idx[8]][8] = f_[4];
        f[idx[7]][7] = f_[5];
        f[idx[3]][3] = f_[6];
        f[idx[5]][5] = f_[7];
        f[idx[4]][4] = f_[8];
    } else {
        for i in 0..9 {
            f[idx[0]][i] = f_[i];
        }
    }

    rho[idx[0]] = r;
    vel[idx[0]] = v;
}

#[rustfmt::skip]
fn collide<const D: usize, const Q: usize>(
    f: &mut VectS<f32, Q>,
    omega: f32,
    rho: f32,
    vel: VectS<f32, D>,
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
) {
    let vv = (vel * vel).sum();

    if (D == 2) && (Q == 9) {
        // Explicitly write out D2Q9 feq calculation

        let v = vel; 
        let vxx = v[0] * v[0];
        let vyy = v[1] * v[1];
        let vxy = v[0] * v[1];

        // 0:        1:       2:       3:       4:      5:      6:       7:      8:
        // [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]
        f[4] -= omega * (f[4] - rho * (2.0 / 9.0) * (2.0 - 3.0 * vv));
        f[7] -= omega * (f[7] - rho * (1.0 / 18.0) * (2.0 + 6.0 * v[0] + 9.0 * vxx - 3.0 * vv));
        f[1] -= omega * (f[1] - rho * (1.0 / 18.0) * (2.0 - 6.0 * v[0] + 9.0 * vxx - 3.0 * vv));
        f[5] -= omega * (f[5] - rho * (1.0 / 18.0) * (2.0 + 6.0 * v[1] + 9.0 * vyy - 3.0 * vv));
        f[3] -= omega * (f[3] - rho * (1.0 / 18.0) * (2.0 - 6.0 * v[1] + 9.0 * vyy - 3.0 * vv));
        f[8] -= omega * (f[8] - rho * (1.0 / 36.0) * (1.0 + 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv));
        f[0] -= omega * (f[0] - rho * (1.0 / 36.0) * (1.0 - 3.0 * (v[0] + v[1]) + 9.0 * vxy + 3.0 * vv));
        f[2] -= omega * (f[2] - rho * (1.0 / 36.0) * (1.0 + 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv));
        f[6] -= omega * (f[6] - rho * (1.0 / 36.0) * (1.0 - 3.0 * (v[1] - v[0]) - 9.0 * vxy + 3.0 * vv));
    } else {
        let f_eq = calc_f_eq(rho, vel, ws, qs);

        // loop-fusion (avoids temporaries and multiple loops over Q)
        for q in 0..Q {
            f[q] -= (f[q] - f_eq[q]) * omega;
        }
    }
}

/// Container for the LBM simulation data which is generic in the dimension and velocity set size.
struct LBM<const D: usize, const Q: usize> {
    ws: VectS<f32, Q>,
    qs: [VectS<f32, D>; Q],
    js: [usize; Q],

    cnt: VectS<isize, D>,

    // upstream indices for each cell
    idx: VectD<VectS<isize, Q>>,

    even: bool,
    f: VectD<VectS<f32, Q>>,

    rho: VectD<f32>,
    vel: VectD<VectS<f32, D>>,
}

impl<const D: usize, const Q: usize> LBM<D, Q> {
    pub fn new(cnt: VectS<usize, D>, ws: [f32; Q], qs: [VectS<f32, D>; Q]) -> LBM<D, Q> {
        let cnt: VectS<isize, D> = cnt.cast();
        let n = cnt.prod() as usize;

        // initialize offset vectors
        let mut idx: VectD<VectS<isize, Q>> = VectD::zeros(n);
        let mut i = 0;
        for sub in Raster::new(cnt) {
            for q in 0..Q {
                let sub_ = sub - qs[q].cast();
                let sub_ = vmod(sub_, cnt);
                let j = sub_to_idx(sub_, cnt);
                idx[i][q] = j;
            }

            i += 1;
        }

        // initialize negative indicies
        // This is a very noddy O(Q^2) and I'm sure we can do better, but Q is small.
        let mut js = [0; Q];
        for i in 0..Q {
            for j in 0..Q {
                let qij = qs[i] + qs[j];
                if (qij * qij).sum() < 1e-8 {
                    js[i] = j;
                    break;
                }
            }
        }

        // Sanity check: all indicies appear in the negative index array.
        for i in 0..Q {
            let mut found = false;
            for j in 0..Q {
                if js[j] == i {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }

        let mut out = LBM {
            ws: VectS::new(ws),
            qs: qs,
            js: js,
            cnt: cnt,
            idx: idx,
            even: true,
            f: VectD::zeros(n),
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
            self.f[i] = calc_f_eq(self.rho[i], self.vel[i], self.ws, self.qs);
        }
    }

    /// Basic implementation of one iteration of the LBM method.
    pub fn step(&mut self, tau: f32) {
        let omega = 1. / tau;
        // TODO: parallelize this!
        for i in 0..self.cnt.prod() {
            if (D == 1) && (Q == 3) {
                update_d1q3_bgk(
                    self.even,
                    omega,
                    &mut self.f,
                    &mut self.rho,
                    &mut self.vel,
                    self.idx[i],
                );
            } else if (D == 2) && (Q == 9) {
                update_d2q9_bgk(
                    self.even,
                    omega,
                    &mut self.f,
                    &mut self.rho,
                    &mut self.vel,
                    self.idx[i],
                );
            } else {
                update_generic_bgk(
                    self.even,
                    omega,
                    &mut self.f,
                    &mut self.rho,
                    &mut self.vel,
                    self.idx[i],
                    self.ws,
                    self.qs,
                    self.js,
                )
            }
        }

        self.even = !self.even;
    }
}

#[wasm_bindgen]
struct D1Q3 {
    lbm: LBM<1, 3>,
}

static D1Q3_Q: [VectS<f32, 1>; 3] = [VectS::new([0.0]), VectS::new([1.0]), VectS::new([-1.0])];
static D1Q3_W: [f32; 3] = [4. / 6., 1. / 6., 1. / 6.];

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
        self.lbm.step(tau);
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
    curl: VectD<f32>,
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
            curl: VectD::zeros(nx * ny),
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
        self.lbm.step(tau);
    }

    pub fn calc_curl(&mut self) {
        let nx = self.lbm.cnt[0];
        let ny = self.lbm.cnt[1];
        for i in 0..nx {
            for j in 0..ny {
                let idx = i + j * nx;
                let idx_xp = fmod(i + 1, nx) + j * nx;
                let idx_xn = fmod(i - 1, nx) + j * nx;
                let idx_yp = i + fmod(j + 1, ny) * nx;
                let idx_yn = i + fmod(j - 1, ny) * nx;

                let dvy_dx = (self.lbm.vel[idx_xp][1] - self.lbm.vel[idx_xn][1]) / 2.0;
                let dvx_dy = (self.lbm.vel[idx_yp][0] - self.lbm.vel[idx_yn][0]) / 2.0;

                self.curl[idx] = dvy_dx - dvx_dy;
            }
        }
    }

    pub fn rho_(&mut self) -> *mut f32 {
        self.lbm.rho.data.as_mut_ptr()
    }

    pub fn vel_(&mut self) -> *mut f32 {
        self.lbm.vel.data.as_mut_ptr() as *mut f32
    }

    pub fn curl_(&mut self) -> *mut f32 {
        self.curl.data.as_mut_ptr()
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
