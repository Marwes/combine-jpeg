// From https://github.com/kaksmet/jpeg-decoder
//
//// Malicious JPEG files can cause operations in the idct to overflow.
// One example is tests/crashtest/images/imagetestsuite/b0b8914cc5f7a6eff409f16d8cc236c5.jpg
// That's why wrapping operators are needed.

use std::num::Wrapping;

// This is based on stb_image's 'stbi__idct_block'.
pub fn dequantize_and_idct_block(
    coefficients: &[i16; 64],
    quantization_table: &[u16; 64],
    output_linestride: usize,
    output: &mut [u8],
) {
    assert!(output_linestride >= 8);

    #[inline(always)]
    fn dequantize(c: i16, q: u16) -> Wrapping<i32> {
        Wrapping(i32::from(c) * i32::from(q))
    }

    let mut temp: [Wrapping<i32>; 64] = unsafe { std::mem::uninitialized() };

    // columns
    for i in 0..8 {
        if (coefficients[i + 8]
            | coefficients[i + 16]
            | coefficients[i + 24]
            | coefficients[i + 32]
            | coefficients[i + 40]
            | coefficients[i + 48]
            | coefficients[i + 56])
            == 0
        {
            let dcterm = dequantize(coefficients[i], quantization_table[i]) << 2;
            temp[i] = dcterm;
            temp[i + 8] = dcterm;
            temp[i + 16] = dcterm;
            temp[i + 24] = dcterm;
            temp[i + 32] = dcterm;
            temp[i + 40] = dcterm;
            temp[i + 48] = dcterm;
            temp[i + 56] = dcterm;
        } else {
            let s0 = dequantize(coefficients[i], quantization_table[i]);
            let s1 = dequantize(coefficients[i + 8], quantization_table[i + 8]);
            let s2 = dequantize(coefficients[i + 16], quantization_table[i + 16]);
            let s3 = dequantize(coefficients[i + 24], quantization_table[i + 24]);
            let s4 = dequantize(coefficients[i + 32], quantization_table[i + 32]);
            let s5 = dequantize(coefficients[i + 40], quantization_table[i + 40]);
            let s6 = dequantize(coefficients[i + 48], quantization_table[i + 48]);
            let s7 = dequantize(coefficients[i + 56], quantization_table[i + 56]);

            let p2 = s2;
            let p3 = s6;
            let p1 = (p2 + p3) * stbi_f2f(0.5411961);
            let t2 = p1 + p3 * stbi_f2f(-1.847759065);
            let t3 = p1 + p2 * stbi_f2f(0.765366865);
            let p2 = s0;
            let p3 = s4;
            let t0 = stbi_fsh(p2 + p3);
            let t1 = stbi_fsh(p2 - p3);
            let x0 = t0 + t3;
            let x3 = t0 - t3;
            let x1 = t1 + t2;
            let x2 = t1 - t2;
            let t0 = s7;
            let t1 = s5;
            let t2 = s3;
            let t3 = s1;
            let p3 = t0 + t2;
            let p4 = t1 + t3;
            let p1 = t0 + t3;
            let p2 = t1 + t2;
            let p5 = (p3 + p4) * stbi_f2f(1.175875602);
            let t0 = t0 * stbi_f2f(0.298631336);
            let t1 = t1 * stbi_f2f(2.053119869);
            let t2 = t2 * stbi_f2f(3.072711026);
            let t3 = t3 * stbi_f2f(1.501321110);
            let p1 = p5 + p1 * stbi_f2f(-0.899976223);
            let p2 = p5 + p2 * stbi_f2f(-2.562915447);
            let p3 = p3 * stbi_f2f(-1.961570560);
            let p4 = p4 * stbi_f2f(-0.390180644);
            let t3 = t3 + p1 + p4;
            let t2 = t2 + p2 + p3;
            let t1 = t1 + p2 + p4;
            let t0 = t0 + p1 + p3;

            // constants scaled things up by 1<<12; let's bring them back
            // down, but keep 2 extra bits of precision
            let x0 = x0 + Wrapping(512);
            let x1 = x1 + Wrapping(512);
            let x2 = x2 + Wrapping(512);
            let x3 = x3 + Wrapping(512);

            temp[i] = (x0 + t3) >> 10;
            temp[i + 56] = (x0 - t3) >> 10;
            temp[i + 8] = (x1 + t2) >> 10;
            temp[i + 48] = (x1 - t2) >> 10;
            temp[i + 16] = (x2 + t1) >> 10;
            temp[i + 40] = (x2 - t1) >> 10;
            temp[i + 24] = (x3 + t0) >> 10;
            temp[i + 32] = (x3 - t0) >> 10;
        }
    }

    assert!(output_linestride >= 8);
    temp.chunks_exact(8)
        .zip(output.chunks_mut(output_linestride))
        .for_each(|(chunk, output_chunk)| {
            // no fast case since the first 1D IDCT spread components out
            let [s0, s1, s2, s3, s4, s5, s6, s7] = *fixed_slice!(chunk; 8);

            let p2 = s2;
            let p3 = s6;
            let p1 = (p2 + p3) * stbi_f2f(0.5411961);
            let t2 = p1 + p3 * stbi_f2f(-1.847759065);
            let t3 = p1 + p2 * stbi_f2f(0.765366865);
            let p2 = s0;
            let p3 = s4;
            let t0 = stbi_fsh(p2 + p3);
            let t1 = stbi_fsh(p2 - p3);
            let x0 = t0 + t3;
            let x3 = t0 - t3;
            let x1 = t1 + t2;
            let x2 = t1 - t2;
            let t0 = s7;
            let t1 = s5;
            let t2 = s3;
            let t3 = s1;
            let p3 = t0 + t2;
            let p4 = t1 + t3;
            let p1 = t0 + t3;
            let p2 = t1 + t2;
            let p5 = (p3 + p4) * stbi_f2f(1.175875602);
            let t0 = t0 * stbi_f2f(0.298631336);
            let t1 = t1 * stbi_f2f(2.053119869);
            let t2 = t2 * stbi_f2f(3.072711026);
            let t3 = t3 * stbi_f2f(1.501321110);
            let p1 = p5 + p1 * stbi_f2f(-0.899976223);
            let p2 = p5 + p2 * stbi_f2f(-2.562915447);
            let p3 = p3 * stbi_f2f(-1.961570560);
            let p4 = p4 * stbi_f2f(-0.390180644);
            let t3 = t3 + p1 + p4;
            let t2 = t2 + p2 + p3;
            let t1 = t1 + p2 + p4;
            let t0 = t0 + p1 + p3;

            // constants scaled things up by 1<<12, plus we had 1<<2 from first
            // loop, plus horizontal and vertical each scale by sqrt(8) so together
            // we've got an extra 1<<3, so 1<<17 total we need to remove.
            // so we want to round that, which means adding 0.5 * 1<<17,
            // aka 65536. Also, we'll end up with -128 to 127 that we want
            // to encode as 0..255 by adding 128, so we'll add that before the shift
            let x0 = x0 + Wrapping(65536 + (128 << 17));
            let x1 = x1 + Wrapping(65536 + (128 << 17));
            let x2 = x2 + Wrapping(65536 + (128 << 17));
            let x3 = x3 + Wrapping(65536 + (128 << 17));

            output_chunk[0] = stbi_clamp((x0 + t3) >> 17);
            output_chunk[7] = stbi_clamp((x0 - t3) >> 17);
            output_chunk[1] = stbi_clamp((x1 + t2) >> 17);
            output_chunk[6] = stbi_clamp((x1 - t2) >> 17);
            output_chunk[2] = stbi_clamp((x2 + t1) >> 17);
            output_chunk[5] = stbi_clamp((x2 - t1) >> 17);
            output_chunk[3] = stbi_clamp((x3 + t0) >> 17);
            output_chunk[4] = stbi_clamp((x3 - t0) >> 17);
        });
}

// take a -128..127 value and stbi__clamp it and convert to 0..255
fn stbi_clamp(Wrapping(x): Wrapping<i32>) -> u8 {
    // trick to use a single test to catch both cases
    if x as u32 > 255 {
        if x < 0 {
            return 0;
        }
        if x > 255 {
            return 255;
        }
    }

    x as u8
}

fn stbi_f2f(x: f32) -> Wrapping<i32> {
    Wrapping((x * 4096.0 + 0.5) as i32)
}

const fn stbi_fsh(x: Wrapping<i32>) -> Wrapping<i32> {
    Wrapping(x.0 << 12)
}
