use std::ops::{Add, Index, IndexMut, Mul, Sub};

use num_traits::{One, Zero};

// S => "static" vector, i.e. size known at compile time
#[derive(Debug, Clone, Copy)]
pub struct VectS<T, const D: usize> {
    data: [T; D],
}

impl<T, const D: usize> VectS<T, D> {
    pub const fn new(data: [T; D]) -> Self {
        Self { data: data }
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
