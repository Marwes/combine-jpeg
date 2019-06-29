use itertools::izip;

use crate::AdobeColorTransform;

pub(crate) type ColorConvertFunc = fn(&mut [u8], &[&[u8]], usize);

pub(crate) fn choose_color_convert_func(
    component_count: usize,
    _is_jfif: bool,
    color_transform: Option<AdobeColorTransform>,
) -> Result<ColorConvertFunc, &'static str> {
    match component_count {
        3 => {
            // http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/JPEG.html#Adobe
            // Unknown means the data is RGB, so we don't need to perform any color conversion on it.
            if color_transform == Some(AdobeColorTransform::Unknown) {
                Ok(color_convert_line_null)
            } else {
                Ok(color_convert_line_ycbcr)
            }
        }
        4 => {
            // http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/JPEG.html#Adobe
            match color_transform {
                Some(AdobeColorTransform::Unknown) => Ok(color_convert_line_cmyk),
                Some(_) => Ok(color_convert_line_ycck),
                None => Err("4 components without Adobe APP14 metadata to tell color space"),
            }
        }
        _ => panic!("Invalid component count {}", component_count),
    }
}

fn color_convert_line_null(_data: &mut [u8], _input: &[&[u8]], _width: usize) {}

fn color_convert_line_ycbcr(data: &mut [u8], input: &[&[u8]], width: usize) {
    let [y, cb, cr] = *fixed_slice!(input; 3);
    let l = y.len().min(cb.len()).min(cr.len()).min(width);

    for (chunk, &y, &cb, &cr) in izip!(
        data.chunks_exact_mut(3).take(width),
        &y[..l],
        &cb[..l],
        &cr[..l]
    ) {
        let chunk = fixed_slice_mut!(chunk; 3);

        let converted = ycbcr_to_rgb(y, cb, cr);

        chunk[0] = converted[0];
        chunk[1] = converted[1];
        chunk[2] = converted[2];
    }
}

fn color_convert_line_ycck(data: &mut [u8], input: &[&[u8]], width: usize) {
    let [y, cb, cr] = *fixed_slice!(input; 3);
    let l = y.len().min(cb.len()).min(cr.len()).min(width);

    for (chunk, &y, &cb, &cr) in izip!(
        data.chunks_exact_mut(4).take(width),
        &y[..l],
        &cb[..l],
        &cr[..l]
    ) {
        let chunk = fixed_slice_mut!(chunk; 4);

        let [r, g, b] = ycbcr_to_rgb(y, cb, cr);
        let k = chunk[3];

        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
        chunk[3] = 255 - k;
    }
}

fn color_convert_line_cmyk(data: &mut [u8], input: &[&[u8]], width: usize) {
    let [c, m, y, k] = *fixed_slice!(input; 4);
    let l = c.len().min(m.len()).min(y.len()).min(k.len()).min(width);

    for (out, &c, &m, &y, &k) in izip!(
        data.chunks_exact_mut(4).take(width),
        &c[..l],
        &m[..l],
        &y[..l],
        &k[..l]
    ) {
        let out = fixed_slice_mut!(out; 4);
        out[0] = 255 - c;
        out[1] = 255 - m;
        out[2] = 255 - y;
        out[3] = 255 - k;
    }
}

macro_rules! call_func {
    ($f: expr, $($expr: expr),*) => {
        [$( $f($expr) ),*]
    }
}
macro_rules! make_table {
    ($name: ident, $func: expr) => {
        make_table!($name, $func => u8)
    };

    ($name: ident, $func: expr => $ty: ty) => {
        #[inline(always)]
        pub fn $name(b: u8) -> $ty {
            const TABLE: [$ty; 256] = call_func!(
                $func, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
                168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
                216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
                232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
                248, 249, 250, 251, 252, 253, 254, 255
            );
            TABLE[usize::from(b)]
        }
    };
}

const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

macro_rules! fix {
    ($e: expr) => {{
        const FIX: i32 = ($e * (1 << 16) as f64 + 0.5) as i32;
        FIX
    }};
}

const fn cr_to_r_fn(i: u8) -> i32 {
    let x = (i as i32) - 128;

    (fix!(1.40200) * x + ONE_HALF) >> SCALEBITS
}
make_table! { cr_to_r, cr_to_r_fn => i32 }

const fn cb_to_b_fn(i: u8) -> i32 {
    let x = (i as i32) - 128;

    (fix!(1.77200) * x + ONE_HALF) >> SCALEBITS
}
make_table! { cb_to_b, cb_to_b_fn => i32 }

const fn cr_to_g_fn(i: u8) -> i32 {
    let x = (i as i32) - 128;

    -fix!(0.71414) * x
}
make_table! { cr_to_g, cr_to_g_fn => i32 }

const fn cb_to_g_fn(i: u8) -> i32 {
    let x = (i as i32) - 128;

    -fix!(0.34414) * x + ONE_HALF
}
make_table! { cb_to_g, cb_to_g_fn => i32 }

// ITU-R BT.601
#[inline(always)]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = y as i32;

    let r = cr_to_r(cr);
    let g = (cb_to_g(cb) + cr_to_g(cr)) >> SCALEBITS;
    let b = cb_to_b(cb);

    #[inline(always)]
    fn clamp(x: i32) -> u8 {
        let i = (x + 256) as usize;
        debug_assert!(i < YCBCR_CLAMP.len());
        // SAFETY Verified by testing all possible `y, cb, cr` combinations
        unsafe { *YCBCR_CLAMP.get_unchecked(i) }
    }
    [clamp(y + r), clamp(y + g), clamp(y + b)]
}

const YCBCR_CLAMP: [u8; 256 * 3] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 40
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 60
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 80
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 100
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 120
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 140
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 160
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 180
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 200
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 220
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 240
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 256
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
    155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
    212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
    231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
    250, 251, 252, 253, 254, 255, // 0
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 20
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 40
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 60
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 80
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 100
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 120
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 140
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 160
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 180
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 200
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 220
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, // 240
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 256
];

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::iproduct;

    fn ycbcr_to_rgb_reference(y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let y = y as f32;
        let cb = cb as f32 - 128.0;
        let cr = cr as f32 - 128.0;

        let r = y + 1.40200 * cr;
        let g = y - 0.34414 * cb - 0.71414 * cr;
        let b = y + 1.77200 * cb;

        [
            clamp((r + 0.5) as i32, 0, 255) as u8,
            clamp((g + 0.5) as i32, 0, 255) as u8,
            clamp((b + 0.5) as i32, 0, 255) as u8,
        ]
    }

    fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
        if value < min {
            return min;
        }
        if value > max {
            return max;
        }
        value
    }

    fn check(y: u8, cb: u8, cr: u8) {
        assert_eq!(
            ycbcr_to_rgb(y, cb, cr),
            ycbcr_to_rgb_reference(y, cb, cr),
            "for {:?}",
            (y, cb, cr)
        );
    }

    #[test]
    fn ycbcr_to_rgb_table_test() {
        check(100, 26, 45);
        check(1, 2, 3);
        check(255, 255, 255);
        check(255, 0, 128);
    }

    #[test]
    fn convert_values() {
        let iter = || {
            (0..=255)
                .map(cr_to_r)
                .chain(
                    iproduct!(0..=255, 0..=255)
                        .map(|(cb, cr)| (cb_to_g(cb) + cr_to_g(cr)) >> SCALEBITS),
                )
                .chain((0..=255).map(cb_to_b))
        };

        let min = iter().min().unwrap();
        assert!(min > -255);
        let max = iter().max().unwrap();
        assert!(max < 255);
    }

    #[test]
    fn check_ycbr_convert_safety() {
        for (y, cb, cr) in iproduct!(0..=255, 0..=255, 0..=255) {
            ycbcr_to_rgb(y, cb, cr);
        }
    }

    #[test]
    fn make_table() {
        const fn two(b: u8) -> u16 {
            b as u16 * 2
        }
        make_table!(test, two => u16);
        assert_eq!(
            (0..=255u16).map(|b| b * 2).collect::<Vec<_>>(),
            (0..=255).map(test).collect::<Vec<_>>(),
        );
    }
}
