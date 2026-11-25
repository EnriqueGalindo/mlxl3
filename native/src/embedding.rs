//! Lossless resident embedding compression; IDs and lookup stay on the GPU.
use crate::array::{Array, Dtype, metal_kernel};
use anyhow::{Context, Result, ensure};

pub enum Embedding {
    Dense(Array),
    Packed {
        shape: [i32; 2],
        main: Array,
        offsets: Array,
        tail: Array,
    },
    /// Lossy: int8 codes with one F16 scale per row (row max-abs / 127). Half the resident bytes of F16.
    Quant8 {
        codes: Array,
        scale: Array,
    },
}

impl Embedding {
    /// Table stored on disk as int8 codes plus one F16 scale per row (`eval/embed_q8.py`).
    pub fn quant8(codes: Array, scale: Array) -> Result<Self> {
        ensure!(
            codes.dtype() == Dtype::Int8
                && codes.shape().len() == 2
                && scale.shape() == [codes.shape()[0], 1],
            "invalid 8-bit embedding tensors"
        );
        let scale = scale.astype_f16()?;
        codes.eval()?;
        scale.eval()?;
        Ok(Self::Quant8 { codes, scale })
    }

    pub fn new(weight: Array, packed: bool) -> Result<Self> {
        if !packed {
            return Ok(Self::Dense(weight));
        }
        ensure!(weight.shape().len() == 2, "embedding must be rank2");
        let shape = [weight.shape()[0], weight.shape()[1]];
        let mut values = weight.pack_embedding()?;
        let tail = values.pop().context("missing embedding tail")?;
        let offsets = values.pop().context("missing embedding offsets")?;
        let main = values.pop().context("missing packed embedding")?;
        let tail_bytes = tail.retained_bytes()?;
        let packed_bytes = main
            .retained_bytes()?
            .checked_add(offsets.retained_bytes()?)
            .and_then(|n| n.checked_add(tail_bytes))
            .context("embedding size overflow")?;
        // Pathological F16 tables need all low bits: preserve the dense fallback.
        if packed_bytes >= weight.retained_bytes()? {
            return Ok(Self::Dense(weight));
        }
        Ok(Self::Packed {
            shape,
            main,
            offsets,
            tail,
        })
    }

    pub fn shape(&self) -> &[i32] {
        match self {
            Self::Dense(a) => a.shape(),
            Self::Packed { shape, .. } => shape,
            Self::Quant8 { codes, .. } => codes.shape(),
        }
    }

    #[cfg(test)]
    pub fn retained_bytes(&self) -> Result<usize> {
        match self {
            Self::Dense(a) => a.retained_bytes(),
            Self::Packed {
                main,
                offsets,
                tail,
                ..
            } => {
                let tail = tail.retained_bytes()?;
                main.retained_bytes()?
                    .checked_add(offsets.retained_bytes()?)
                    .and_then(|n| n.checked_add(tail))
                    .context("packed embedding size overflow")
            }
            Self::Quant8 { codes, scale } => codes
                .retained_bytes()?
                .checked_add(scale.retained_bytes()?)
                .context("quantized embedding size overflow"),
        }
    }

    // Private model callers guarantee vocabulary bounds, including device IDs.
    pub fn take(&self, ids: &Array, axis: i32) -> Result<Array> {
        match self {
            Self::Dense(a) => a.take(ids, axis),
            Self::Quant8 { codes, scale } => {
                ensure!(axis == 0, "invalid quantized embedding axis");
                codes.take(ids, 0)?.astype_f16()?.mul(&scale.take(ids, 0)?)
            }
            Self::Packed {
                shape,
                main,
                offsets,
                tail,
            } => {
                ensure!(
                    axis == 0 && matches!(ids.dtype(), Dtype::Int32 | Dtype::UInt32),
                    "invalid packed embedding IDs"
                );
                let mut output_shape = ids.shape().to_vec();
                output_shape.push(shape[1]);
                let count = output_shape
                    .iter()
                    .try_fold(1i32, |n, &d| n.checked_mul(d))
                    .filter(|&n| n > 0)
                    .context("packed embedding output is empty or too large")?;
                let header = format!(
                    "using namespace metal;\n#define WIDTH {}u\n#define VOCAB {}u\n#define OUTPUT_COUNT {}u\n",
                    shape[1], shape[0], count
                );
                metal_kernel(
                    "mlxl3_packed_embedding",
                    &["main", "offsets", "tail", "ids"],
                    &["output"],
                    &header,
                    include_str!("../shaders/packed_embedding.metal"),
                    &[main, offsets, tail, ids],
                    &[output_shape],
                    &[Dtype::Float16],
                    [count, 1, 1],
                    [256, 1, 1],
                )?
                .pop()
                .context("missing packed embedding output")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "physical Apple GPU; run alone"]
    fn packed_embeddings_preserve_all_f16_patterns_and_permutations() -> Result<()> {
        // Mix all bit patterns with enough BF16-derived values to exercise packing.
        let mut bits = (0..65536).map(|x| x as u16).collect::<Vec<_>>();
        bits.extend((0..65536).map(|x| (x as u16) & !7));
        let dense = Array::from_f16_bits(&bits, &[1024, 128])?;
        let packed = Embedding::new(dense.try_clone()?, true)?;
        assert!(matches!(packed, Embedding::Packed { .. }));
        for reverse in [false, true] {
            let ids = (0..1024)
                .map(|x| if reverse { 1023 - x } else { x })
                .collect::<Vec<i32>>();
            let id_array = Array::from_i32(&ids, &[32, 32])?;
            let actual = packed.take(&id_array, 0)?;
            assert_eq!(actual.shape(), &[32, 32, 128]);
            let expected = ids
                .iter()
                .flat_map(|&id| {
                    bits[id as usize * 128..(id as usize + 1) * 128]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            assert_eq!(actual.to_f16_bits()?, expected);
        }
        assert!(Embedding::new(Array::zeros_dtype(&[1, 127], Dtype::Float16)?, true).is_err());
        assert!(Embedding::new(Array::zeros(&[1, 128])?, true).is_err());
        assert!(packed.take(&Array::from_f32(&[1.], &[1])?, 0).is_err());
        assert!(packed.take(&Array::from_i32(&[1], &[1])?, 1).is_err());
        let all_low = Embedding::new(Array::from_f16_bits(&[1; 128], &[1, 128])?, true)?;
        assert!(
            matches!(all_low, Embedding::Dense(_)),
            "unprofitable packing must fall back"
        );
        Ok(())
    }
}
