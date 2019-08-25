use crate::{Component, ComponentVec, Error, Result, UnsupportedFeature};

use itertools::izip;

pub(crate) struct Upsampler {
    components: ComponentVec<UpsamplerComponent>,
}

struct UpsamplerComponent {
    upsampler: Box<dyn Upsample + Sync>,
    width: usize,
    height: usize,
    row_stride: usize,
}

impl Upsampler {
    pub fn new(
        components: &[Component],
        output_width: u16,
        output_height: u16,
    ) -> Result<Upsampler> {
        let h_max = components
            .iter()
            .map(|c| c.horizontal_sampling_factor)
            .max()
            .unwrap();
        let v_max = components
            .iter()
            .map(|c| c.vertical_sampling_factor)
            .max()
            .unwrap();
        Ok(Upsampler {
            components: components
                .iter()
                .map(|component| {
                    let upsampler = choose_upsampler(
                        (
                            component.horizontal_sampling_factor,
                            component.vertical_sampling_factor,
                        ),
                        (h_max, v_max),
                        output_width,
                        output_height,
                    )?;
                    Ok(UpsamplerComponent {
                        upsampler: upsampler,
                        width: component.size.width as usize,
                        height: component.size.height as usize,
                        row_stride: component.block_size.width as usize * 8,
                    })
                })
                .collect::<Result<_>>()?,
        })
    }

    pub fn upsample_and_interleave_row<'s>(
        &'s self,
        line_buffers: &'s mut ComponentVec<Vec<u8>>,
        component_data: &'s [Vec<u8>],
        row: usize,
        output_width: usize,
    ) -> impl Iterator<Item = &'s [u8]> + 's {
        let component_count = component_data.len();

        debug_assert_eq!(component_count, self.components.len());

        izip!(&self.components, component_data, line_buffers).map(
            move |(component, data, line_buffer)| {
                component.upsampler.upsample_row(
                    data,
                    component.width,
                    component.height,
                    component.row_stride,
                    row,
                    output_width,
                    line_buffer,
                )
            },
        )
    }
}

struct UpsamplerH1V1;
struct UpsamplerH2V1;
struct UpsamplerH1V2;
struct UpsamplerH2V2;

struct UpsamplerGeneric {
    horizontal_scaling_factor: u8,
    vertical_scaling_factor: u8,
}

fn choose_upsampler(
    sampling_factors: (u8, u8),
    max_sampling_factors: (u8, u8),
    output_width: u16,
    output_height: u16,
) -> Result<Box<dyn Upsample + Sync>> {
    let h1 = sampling_factors.0 == max_sampling_factors.0 || output_width == 1;
    let v1 = sampling_factors.1 == max_sampling_factors.1 || output_height == 1;
    let h2 = sampling_factors.0 * 2 == max_sampling_factors.0;
    let v2 = sampling_factors.1 * 2 == max_sampling_factors.1;

    if h1 && v1 {
        Ok(Box::new(UpsamplerH1V1))
    } else if h2 && v1 {
        Ok(Box::new(UpsamplerH2V1))
    } else if h1 && v2 {
        Ok(Box::new(UpsamplerH1V2))
    } else if h2 && v2 {
        Ok(Box::new(UpsamplerH2V2))
    } else {
        if max_sampling_factors.0 % sampling_factors.0 != 0
            || max_sampling_factors.1 % sampling_factors.1 != 0
        {
            Err(Error::Unsupported(
                UnsupportedFeature::NonIntegerSubsamplingRatio,
            ))
        } else {
            Ok(Box::new(UpsamplerGeneric {
                horizontal_scaling_factor: max_sampling_factors.0 / sampling_factors.0,
                vertical_scaling_factor: max_sampling_factors.1 / sampling_factors.1,
            }))
        }
    }
}

trait Upsample: Send + Sync {
    fn upsample_row<'s>(
        &self,
        input: &'s [u8],
        input_width: usize,
        input_height: usize,
        row_stride: usize,
        row: usize,
        output_width: usize,
        output: &'s mut Vec<u8>,
    ) -> &'s [u8];
}

impl Upsample for UpsamplerH1V1 {
    fn upsample_row<'s>(
        &self,
        input: &'s [u8],
        _input_width: usize,
        _input_height: usize,
        row_stride: usize,
        row: usize,
        output_width: usize,
        _output: &'s mut Vec<u8>,
    ) -> &'s [u8] {
        let input = &input[row * row_stride..];
        &input[..output_width]
    }
}

impl Upsample for UpsamplerH2V1 {
    fn upsample_row<'s>(
        &self,
        input: &'s [u8],
        input_width: usize,
        _input_height: usize,
        row_stride: usize,
        row: usize,
        output_width: usize,
        output: &'s mut Vec<u8>,
    ) -> &'s [u8] {
        output.resize(output_width, 0);

        let input = &input[row * row_stride..];

        if input_width == 1 {
            output[0] = input[0];
            output[1] = input[0];
            return output;
        }

        output[0] = input[0];
        output[1] = ((u32::from(input[0]) * 3 + u32::from(input[1]) + 2) >> 2) as u8;

        for (out, in_) in output[2..].chunks_exact_mut(2).zip(input.windows(3)) {
            let sample = 3 * u32::from(in_[1]) + 2;
            out[0] = ((sample + u32::from(in_[0])) >> 2) as u8;
            out[1] = ((sample + u32::from(in_[2])) >> 2) as u8;
        }

        output[(input_width - 1) * 2] =
            ((u32::from(input[input_width - 1]) * 3 + u32::from(input[input_width - 2]) + 2) >> 2)
                as u8;
        output[(input_width - 1) * 2 + 1] = input[input_width - 1];

        output
    }
}

impl Upsample for UpsamplerH1V2 {
    fn upsample_row<'s>(
        &self,
        input: &'s [u8],
        _input_width: usize,
        input_height: usize,
        row_stride: usize,
        row: usize,
        output_width: usize,
        output: &'s mut Vec<u8>,
    ) -> &'s [u8] {
        output.resize(output_width, 0);

        debug_assert!(output.len() == output_width); // TODO Remove output_width

        let row_near = row as f32 / 2.0;
        // If row_near's fractional is 0.0 we want row_far to be the previous row and if it's 0.5 we
        // want it to be the next row.
        let row_far = (row_near + row_near.fract() * 3.0 - 0.25).min((input_height - 1) as f32);

        let input_near = &input[row_near as usize * row_stride..];
        let input_far = &input[row_far as usize * row_stride..];

        for ((out, &near), &far) in output.iter_mut().zip(input_near).zip(input_far) {
            *out = ((3 * u32::from(near) as u32 + u32::from(far) + 2) >> 2) as u8;
        }

        output
    }
}

impl Upsample for UpsamplerH2V2 {
    fn upsample_row<'s>(
        &self,
        input: &'s [u8],
        input_width: usize,
        input_height: usize,
        row_stride: usize,
        row: usize,
        _output_width: usize,
        output: &'s mut Vec<u8>,
    ) -> &'s [u8] {
        output.resize(input_width * 2, 0);

        let row_near = row as f32 / 2.0;
        // If row_near's fractional is 0.0 we want row_far to be the previous row and if it's 0.5 we
        // want it to be the next row.
        let row_far = (row_near + row_near.fract() * 3.0 - 0.25).min((input_height - 1) as f32);

        let input_near = &input[row_near as usize * row_stride..];
        let input_far = &input[row_far as usize * row_stride..];

        if input_width == 1 {
            let value = ((3 * u32::from(input_near[0]) + u32::from(input_far[0]) + 2) >> 2) as u8;
            output[0] = value;
            output[1] = value;
            return output;
        }

        let mut t1 = 3 * u32::from(input_near[0]) + u32::from(input_far[0]);
        output[0] = ((t1 + 2) >> 2) as u8;

        for ((out, &near), &far) in output[1..]
            .chunks_exact_mut(2)
            .zip(&input_near[1..])
            .zip(&input_far[1..])
        {
            let t0 = t1;
            t1 = 3 * u32::from(near) + u32::from(far);

            out[0] = ((3 * t0 + t1 + 8) >> 4) as u8;
            out[1] = ((3 * t1 + t0 + 8) >> 4) as u8;
        }

        output[input_width * 2 - 1] = ((t1 + 2) >> 2) as u8;

        output
    }
}

impl Upsample for UpsamplerGeneric {
    // Uses nearest neighbor sampling
    fn upsample_row<'s>(
        &self,
        input: &'s [u8],
        input_width: usize,
        _input_height: usize,
        row_stride: usize,
        row: usize,
        output_width: usize,
        output: &'s mut Vec<u8>,
    ) -> &'s [u8] {
        output.resize(output_width, 0);

        let start = (row / self.vertical_scaling_factor as usize) * row_stride;
        let input = &input[start..(start + input_width)];
        for (out_chunk, val) in output
            .chunks_exact_mut(usize::from(self.horizontal_scaling_factor))
            .zip(input)
        {
            for out in out_chunk {
                *out = *val;
            }
        }

        output
    }
}
