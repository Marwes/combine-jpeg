use crate::AdobeColorTransform;

pub(crate) fn choose_color_convert_func(
    component_count: usize,
    _is_jfif: bool,
    color_transform: Option<AdobeColorTransform>,
) -> Result<fn(&mut [u8], usize), &'static str> {
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

fn color_convert_line_null(_data: &mut [u8], _width: usize) {}

fn color_convert_line_ycbcr(data: &mut [u8], width: usize) {
    for i in 0..width {
        let (r, g, b) = ycbcr_to_rgb(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);

        data[i * 3] = r;
        data[i * 3 + 1] = g;
        data[i * 3 + 2] = b;
    }
}

fn color_convert_line_ycck(data: &mut [u8], width: usize) {
    for i in 0..width {
        let (r, g, b) = ycbcr_to_rgb(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
        let k = data[i * 4 + 3];

        data[i * 4] = r;
        data[i * 4 + 1] = g;
        data[i * 4 + 2] = b;
        data[i * 4 + 3] = 255 - k;
    }
}

fn color_convert_line_cmyk(data: &mut [u8], width: usize) {
    for i in 0..width {
        data[i * 4] = 255 - data[i * 4];
        data[i * 4 + 1] = 255 - data[i * 4 + 1];
        data[i * 4 + 2] = 255 - data[i * 4 + 2];
        data[i * 4 + 3] = 255 - data[i * 4 + 3];
    }
}

// ITU-R BT.601
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as f32;
    let cb = cb as f32 - 128.0;
    let cr = cr as f32 - 128.0;

    let r = y + 1.40200 * cr;
    let g = y - 0.34414 * cb - 0.71414 * cr;
    let b = y + 1.77200 * cb;

    (
        clamp((r + 0.5) as i32, 0, 255) as u8,
        clamp((g + 0.5) as i32, 0, 255) as u8,
        clamp((b + 0.5) as i32, 0, 255) as u8,
    )
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