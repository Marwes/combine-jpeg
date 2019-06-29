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

// ITU-R BT.601
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
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
