use std::ops::{Index, IndexMut};

use num_traits::{One, Zero};

// D => "dynamic", i.e. size known at run-time
// Wraps a std::Vec but gives us some convenience indexing.
#[derive(Debug, Clone)]
pub struct VectD<T> {
    pub data: Vec<T>,
}

impl<T> VectD<T> {
    pub fn len(&self) -> isize {
        self.data.len() as isize
    }

    pub fn zeros(size: usize) -> VectD<T>
    where
        T: Zero + Copy,
    {
        Self {
            data: vec![T::zero(); size],
        }
    }

    pub fn ones(size: usize) -> VectD<T>
    where
        T: One + Copy,
    {
        Self {
            data: vec![T::one(); size],
        }
    }

    pub fn map<S>(self, func: fn(T) -> S) -> VectD<S> {
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

        return unsafe { &self.data.get_unchecked(index as usize) };
    }
}

impl<T> IndexMut<isize> for VectD<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        let index = if index >= 0 {
            index
        } else {
            self.data.len() as isize + index
        };

        return unsafe { self.data.get_unchecked_mut(index as usize) };
    }
}
