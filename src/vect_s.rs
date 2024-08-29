use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use num_traits::{AsPrimitive, One, Zero};

// S => "static" vector, i.e. size known at compile time
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectS<T, const D: usize> {
    data: [T; D],
}

impl<T, const D: usize> VectS<T, D> {
    pub const fn new(data: [T; D]) -> Self {
        Self { data: data }
    }

    pub fn cast<S>(&self) -> VectS<S, D>
    where
        T: AsPrimitive<S>,
        S: Copy + 'static,
    {
        VectS::<S, D> {
            data: self.data.map(|x| x.as_()),
        }
    }

    pub fn map<S>(&self, func: impl Fn(T) -> S) -> VectS<S, D>
    where
        T: Copy,
        S: Copy + 'static,
    {
        VectS {
            data: self.data.map(func),
        }
    }

    pub fn map_with_idx<S>(&self, func: impl Fn(usize, T) -> S) -> VectS<S, D>
    where
        T: Copy,
        S: Copy + Default + 'static,
    {
        let mut out = VectS::new([S::default(); D]);
        for i in 0..D {
            out.data[i] = func(i, self.data[i]);
        }
        out
    }

    pub fn sum(&self) -> T
    where
        T: Zero + Add<Output = T> + Copy,
    {
        let mut out = T::zero();
        for d in 0..D {
            out = out + self[d];
        }
        return out;
    }

    pub fn prod(&self) -> T
    where
        T: One + Mul<T> + Copy,
    {
        let mut out = T::one();
        for i in 0..D {
            out = out * self[i];
        }
        out
    }

    pub fn cumprod(&self) -> Self
    where
        T: One + Mul<Output = T> + Copy,
    {
        let mut out = Self::one();
        out[0] = self[0];
        for i in 1..D {
            out[i] = out[i - 1] * self[i];
        }
        out
    }

    pub fn cumsum(&self) -> Self
    where
        T: One + Add<Output = T> + Copy,
    {
        let mut out = Self::one();
        out[0] = self[0];
        for i in 1..D {
            out[i] = out[i - 1] + self[i];
        }
        out
    }
}

impl<T, const D: usize> Default for VectS<T, D>
where
    T: Zero + Copy,
{
    fn default() -> Self {
        Self {
            data: [T::zero(); D],
        }
    }
}

impl<T, const D: usize> Zero for VectS<T, D>
where
    T: Zero + Copy,
{
    fn zero() -> Self {
        return Self {
            data: [T::zero(); D],
        };
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|d| d.is_zero())
    }
}

impl<T, const D: usize> One for VectS<T, D>
where
    T: One + Copy,
{
    fn one() -> Self {
        return Self {
            data: [T::one(); D],
        };
    }
}

impl<T, const D: usize> Index<usize> for VectS<T, D> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[index];
    }
}

impl<T, const D: usize> IndexMut<usize> for VectS<T, D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.data[index];
    }
}

impl<T, const D: usize> Add for VectS<T, D>
where
    T: Zero + Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = Self::zero();
        for d in 0..D {
            out[d] = self[d] + rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Sub for VectS<T, D>
where
    T: Zero + Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = Self::zero();
        for d in 0..D {
            out[d] = self[d] - rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Mul for VectS<T, D>
where
    T: One + Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = Self::one();
        for d in 0..D {
            out[d] = self[d] * rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Div for VectS<T, D>
where
    T: One + Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut out = Self::one();
        for d in 0..D {
            out[d] = self[d] / rhs[d];
        }
        return out;
    }
}

impl<T, const D: usize> Mul<T> for VectS<T, D>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            data: self.data.map(|x| x * rhs),
        }
    }
}

impl<T, const D: usize> Div<T> for VectS<T, D>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {
            data: self.data.map(|x| x / rhs),
        }
    }
}

/// impl for concrete f32 rather than T, since rust does not allow the generic one :(
impl<const D: usize> Mul<VectS<f32, D>> for f32 {
    type Output = VectS<f32, D>;

    fn mul(self, rhs: VectS<f32, D>) -> Self::Output {
        rhs * self
    }
}

/// impl for concrete f32 rather than T, since rust does not allow the generic one :(
impl<const D: usize> Div<VectS<f32, D>> for f32 {
    type Output = VectS<f32, D>;

    fn div(self, rhs: VectS<f32, D>) -> Self::Output {
        rhs / self
    }
}
