// From https://github.com/kaksmet/jpeg-decoder
//
//// Malicious JPEG files can cause operations in the idct to overflow.
// One example is tests/crashtest/images/imagetestsuite/b0b8914cc5f7a6eff409f16d8cc236c5.jpg
// That's why wrapping operators are needed.

use std::{
    mem::{self, MaybeUninit},
    num::Wrapping,
};

use crate::clamp::stbi_clamp;

// This is based on stb_image's 'stbi__idct_block'.
pub fn dequantize_and_idct_block<'a, I>(
    coefficients: &[i16; 64],
    quantization_table: &[u16; 64],
    output: I,
) where
    I: IntoIterator<Item = &'a mut [u8; 8]>,
    I::IntoIter: ExactSizeIterator<Item = &'a mut [u8; 8]>,
{
    #[inline(always)]
    fn dequantize(c: i16, q: u16) -> Wrapping<i32> {
        Wrapping(i32::from(c) * i32::from(q))
    }

    let output = output.into_iter();
    debug_assert!(
        output.len() >= 8,
        "Output iterator has the wrong length: {}",
        output.len()
    );

    // SAFETY This gets fully initialized in the columns loop but since it iterates over columns
    // LLVM does not realize this and elide the initialization
    let mut temp: [MaybeUninit<Wrapping<i32>>; 64] = [MaybeUninit::uninit(); 64];

    // columns
    for i in 0..8 {
        if coefficients[i + 8] == 0
            && coefficients[i + 16] == 0
            && coefficients[i + 24] == 0
            && coefficients[i + 32] == 0
            && coefficients[i + 40] == 0
            && coefficients[i + 48] == 0
            && coefficients[i + 56] == 0
        {
            let dcterm = dequantize(coefficients[i], quantization_table[i]) << 2;
            unsafe {
                temp[i].as_mut_ptr().write(dcterm);
                temp[i + 8].as_mut_ptr().write(dcterm);
                temp[i + 16].as_mut_ptr().write(dcterm);
                temp[i + 24].as_mut_ptr().write(dcterm);
                temp[i + 32].as_mut_ptr().write(dcterm);
                temp[i + 40].as_mut_ptr().write(dcterm);
                temp[i + 48].as_mut_ptr().write(dcterm);
                temp[i + 56].as_mut_ptr().write(dcterm);
            }
        } else {
            let s0 = dequantize(coefficients[i], quantization_table[i]);
            let s1 = dequantize(coefficients[i + 8], quantization_table[i + 8]);
            let s2 = dequantize(coefficients[i + 16], quantization_table[i + 16]);
            let s3 = dequantize(coefficients[i + 24], quantization_table[i + 24]);
            let s4 = dequantize(coefficients[i + 32], quantization_table[i + 32]);
            let s5 = dequantize(coefficients[i + 40], quantization_table[i + 40]);
            let s6 = dequantize(coefficients[i + 48], quantization_table[i + 48]);
            let s7 = dequantize(coefficients[i + 56], quantization_table[i + 56]);

            let Kernel {
                xs: [x0, x1, x2, x3],
                ts: [t0, t1, t2, t3],
            } = kernel(
                [s0, s1, s2, s3, s4, s5, s6, s7],
                // constants scaled things up by 1<<12; let's bring them back
                // down, but keep 2 extra bits of precision
                512,
            );

            unsafe {
                temp[i].as_mut_ptr().write((x0 + t3) >> 10);
                temp[i + 56].as_mut_ptr().write((x0 - t3) >> 10);
                temp[i + 8].as_mut_ptr().write((x1 + t2) >> 10);
                temp[i + 48].as_mut_ptr().write((x1 - t2) >> 10);
                temp[i + 16].as_mut_ptr().write((x2 + t1) >> 10);
                temp[i + 40].as_mut_ptr().write((x2 - t1) >> 10);
                temp[i + 24].as_mut_ptr().write((x3 + t0) >> 10);
                temp[i + 32].as_mut_ptr().write((x3 - t0) >> 10);
            }
        }
    }

    // SAFTY `temp` was initialized by the previous loop
    let temp = unsafe {
        mem::transmute::<&mut [MaybeUninit<Wrapping<i32>>; 64], &mut [Wrapping<i32>; 64]>(&mut temp)
    };

    for (chunk, output_chunk) in temp.chunks_exact(8).zip(output) {
        let chunk = fixed_slice!(chunk; 8);

        // constants scaled things up by 1<<12, plus we had 1<<2 from first
        // loop, plus horizontal and vertical each scale by sqrt(8) so together
        // we've got an extra 1<<3, so 1<<17 total we need to remove.
        // so we want to round that, which means adding 0.5 * 1<<17,
        // aka 65536. Also, we'll end up with -128 to 127 that we want
        // to encode as 0..255 by adding 128, so we'll add that before the shift
        const X_SCALE: i32 = 65536 + (128 << 17);

        let [s0, s1, s2, s3, s4, s5, s6, s7] = *chunk;
        if s1.0 == 0 && s2.0 == 0 && s3.0 == 0 && s4.0 == 0 && s5.0 == 0 && s6.0 == 0 && s7.0 == 0 {
            let dcterm = stbi_clamp((stbi_fsh(s0) + Wrapping(X_SCALE)) >> 17);
            output_chunk[0] = dcterm;
            output_chunk[1] = dcterm;
            output_chunk[2] = dcterm;
            output_chunk[3] = dcterm;
            output_chunk[4] = dcterm;
            output_chunk[5] = dcterm;
            output_chunk[6] = dcterm;
            output_chunk[7] = dcterm;
        } else {
            let Kernel {
                xs: [x0, x1, x2, x3],
                ts: [t0, t1, t2, t3],
            } = kernel(*chunk, X_SCALE);

            output_chunk[0] = stbi_clamp((x0 + t3) >> 17);
            output_chunk[7] = stbi_clamp((x0 - t3) >> 17);
            output_chunk[1] = stbi_clamp((x1 + t2) >> 17);
            output_chunk[6] = stbi_clamp((x1 - t2) >> 17);
            output_chunk[2] = stbi_clamp((x2 + t1) >> 17);
            output_chunk[5] = stbi_clamp((x2 - t1) >> 17);
            output_chunk[3] = stbi_clamp((x3 + t0) >> 17);
            output_chunk[4] = stbi_clamp((x3 - t0) >> 17);
        }
    }
}

struct Kernel {
    xs: [Wrapping<i32>; 4],
    ts: [Wrapping<i32>; 4],
}

#[inline]
fn kernel_x([s0, s2, s4, s6]: [Wrapping<i32>; 4], x_scale: i32) -> [Wrapping<i32>; 4] {
    // Even `chunk` indicies
    let (t2, t3);
    {
        let p2 = s2;
        let p3 = s6;

        let p1 = (p2 + p3) * stbi_f2f(0.5411961);
        t2 = p1 + p3 * stbi_f2f(-1.847759065);
        t3 = p1 + p2 * stbi_f2f(0.765366865);
    }

    let (t0, t1);
    {
        let p2 = s0;
        let p3 = s4;

        t0 = stbi_fsh(p2 + p3);
        t1 = stbi_fsh(p2 - p3);
    }

    let x0 = t0 + t3;
    let x3 = t0 - t3;
    let x1 = t1 + t2;
    let x2 = t1 - t2;

    let x_scale = Wrapping(x_scale);

    [x0 + x_scale, x1 + x_scale, x2 + x_scale, x3 + x_scale]
}

#[inline]
fn kernel_t([s1, s3, s5, s7]: [Wrapping<i32>; 4]) -> [Wrapping<i32>; 4] {
    // Odd `chunk` indicies
    let mut t0 = s7;
    let mut t1 = s5;
    let mut t2 = s3;
    let mut t3 = s1;

    let p3 = t0 + t2;
    let p4 = t1 + t3;
    let p1 = t0 + t3;
    let p2 = t1 + t2;
    let p5 = (p3 + p4) * stbi_f2f(1.175875602);

    t0 *= stbi_f2f(0.298631336);
    t1 *= stbi_f2f(2.053119869);
    t2 *= stbi_f2f(3.072711026);
    t3 *= stbi_f2f(1.501321110);

    let p1 = p5 + p1 * stbi_f2f(-0.899976223);
    let p2 = p5 + p2 * stbi_f2f(-2.562915447);
    let p3 = p3 * stbi_f2f(-1.961570560);
    let p4 = p4 * stbi_f2f(-0.390180644);

    t3 += p1 + p4;
    t2 += p2 + p3;
    t1 += p2 + p4;
    t0 += p1 + p3;

    [t0, t1, t2, t3]
}

#[inline]
fn kernel([s0, s1, s2, s3, s4, s5, s6, s7]: [Wrapping<i32>; 8], x_scale: i32) -> Kernel {
    Kernel {
        xs: kernel_x([s0, s2, s4, s6], x_scale),
        ts: kernel_t([s1, s3, s5, s7]),
    }
}

#[inline]
fn stbi_f2f(x: f32) -> Wrapping<i32> {
    Wrapping((x * 4096.0 + 0.5) as i32)
}

#[inline]
const fn stbi_fsh(x: Wrapping<i32>) -> Wrapping<i32> {
    Wrapping(x.0 << 12)
}
