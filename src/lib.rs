use std::mem;

use arrayvec::ArrayVec;

use derive_more::{Display, From};

use combine::{
    easy,
    error::{Consumed, ParseError, StreamError},
    parser,
    parser::{
        byte::num::be_u16,
        item::{any, eof, satisfy_map, value},
        range::{range, take},
        repeat::{count_min_max, many1, skip_many1},
    },
    stream::{FullRangeStream, StreamErrorFor},
    Parser,
};

use marker::{marker, Marker};

mod color_conversion;
mod huffman;
mod idct;
mod marker;
mod upsampler;

#[derive(Debug, PartialEq, Eq, Display)]
pub enum UnsupportedFeature {
    NonIntegerSubsamplingRatio,
}

#[derive(Debug, PartialEq, Display, From)]
pub enum Error {
    #[display(fmt = "{}", _0)]
    Unsupported(UnsupportedFeature),

    #[display(fmt = "{}", _0)]
    Parse(easy::Errors<String, String, usize>),

    #[display(fmt = "{}", _0)]
    Message(&'static str),
}

type Result<T, E = Error> = std::result::Result<T, E>;

fn split_4_bit<'a, I>() -> impl Parser<Output = (u8, u8), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b >> 4, b & 0x0F))
}

#[derive(Copy, Clone, Debug)]
struct Segment<'a> {
    marker: Marker,
    data: &'a [u8],
}

fn segment<'a, I>() -> impl Parser<Output = Segment<'a>, Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    marker().then_partial(|&mut marker| {
        match marker {
        Marker::SOI | Marker::RST(_) | Marker::EOI => {
            Parser::left(value(Segment { marker, data: &[] }).left())
        }
        Marker::DRI => Parser::left(take(4).map(move |data| Segment { marker, data }).right()),
        Marker::SOF(_) | Marker::DHT | Marker::DQT | Marker::SOS | Marker::APP(_) | Marker::COM => be_u16() // TODO Check length >= 2
            .then(|quantization_table_len| take((quantization_table_len - 2).into()))
            .map(move |data| Segment { marker, data })
            .right(),
    }
    })
}

#[derive(Default, Copy, Clone)]
pub struct Dimensions {
    pub width: u16,
    pub height: u16,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum CodingProcess {
    Baseline,
    Sequential,
    Progressive,
    LossLess,
}

type Components = ArrayVec<[Component; 256]>;

pub struct Frame {
    pub marker_index: u8,
    pub precision: u8,
    pub lines: u16,
    pub samples_per_line: u16,
    pub mcu_size: Dimensions,
    pub components: Components,
}

impl Frame {
    fn coding_process(&self) -> CodingProcess {
        match self.marker_index {
            0 => CodingProcess::Baseline,
            1 | 5 | 9 | 13 => CodingProcess::Sequential,
            2 | 6 | 10 | 14 => CodingProcess::Progressive,
            3 | 7 | 11 | 15 => CodingProcess::LossLess,
            i => panic!("Unknown SOF marker {}", i),
        }
    }

    fn component_width(&self, index: usize) -> u16 {
        self.mcu_size.width * u16::from(self.components[index].horizontal_sampling_factor)
    }
}

#[derive(Copy, Clone, Default)]
pub struct Component {
    pub component_identifier: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
    pub quantization_table_destination_selector: u8,

    // Computed parts
    pub block_size: Dimensions,
    pub size: Dimensions,
}

fn sof<'a, I>(marker_index: u8) -> impl Parser<Output = Frame, Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    (any(), be_u16(), be_u16(), any()).then_partial(
        move |&mut (precision, lines, samples_per_line, components_in_frame)| {
            let component = (any(), split_4_bit(), any()).map(
                |(
                    component_identifier,
                    (horizontal_sampling_factor, vertical_sampling_factor),
                    quantization_table_destination_selector,
                )| Component {
                    component_identifier,
                    horizontal_sampling_factor,
                    vertical_sampling_factor,
                    quantization_table_destination_selector,

                    // Filled in later
                    block_size: Default::default(),
                    size: Default::default(),
                },
            );
            count_min_max(
                usize::from(components_in_frame),
                usize::from(components_in_frame),
                component,
            )
            .map(move |mut components: Components| {
                let mcu_size;
                {
                    let h_max = f32::from(
                        components
                            .iter()
                            .map(|c| c.horizontal_sampling_factor)
                            .max()
                            .unwrap(),
                    );
                    let v_max = f32::from(
                        components
                            .iter()
                            .map(|c| c.vertical_sampling_factor)
                            .max()
                            .unwrap(),
                    );

                    let samples_per_line = f32::from(samples_per_line);
                    let lines = f32::from(lines);

                    mcu_size = Dimensions {
                        width: (samples_per_line / (h_max * 8.0)).ceil() as u16,
                        height: (lines / (v_max * 8.0)).ceil() as u16,
                    };

                    for component in &mut components[..] {
                        component.size.width = (f32::from(samples_per_line)
                            * (f32::from(component.horizontal_sampling_factor) / h_max))
                            .ceil() as u16;
                        component.size.height = (lines
                            * (f32::from(component.vertical_sampling_factor) / v_max))
                            .ceil() as u16;

                        component.block_size.width =
                            mcu_size.width * u16::from(component.horizontal_sampling_factor);
                        component.block_size.height =
                            mcu_size.height * u16::from(component.vertical_sampling_factor);
                    }
                }

                Frame {
                    marker_index,
                    precision,
                    lines,
                    samples_per_line,
                    mcu_size,
                    components,
                }
            })
        },
    )
}

macro_rules! fixed_slice {
    ($expr: expr; $len: tt) => {{
        unsafe fn transmute_array<T>(xs: &[T]) -> &[T; $len] {
            assert!(xs.len() == $len);
            &*(xs.as_ptr() as *const [T; $len])
        }
        unsafe { transmute_array($expr) }
    }};
}

struct DHT<'a> {
    table_class: huffman::TableClass,
    destination: u8,
    code_lengths: &'a [u8; 16],
    values: &'a [u8],
}

fn huffman_table<'a, I>() -> impl Parser<Output = DHT<'a>, Input = I> + 'a
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    (
        split_4_bit().and_then(
            |(table_class, destination)| -> Result<_, StreamErrorFor<I>> {
                Ok((
                    huffman::TableClass::new(table_class).ok_or_else(|| {
                        StreamErrorFor::<I>::message_static_message("Invalid huffman table class")
                    })?,
                    destination,
                ))
            },
        ),
        take(16).map(|xs| fixed_slice!(xs; 16)),
    )
        .then_partial(
            |&mut ((table_class, destination), code_lengths): &mut (_, &'a [u8; 16])| {
                take(code_lengths.iter().map(|&x| usize::from(x)).sum()).map(move |values| DHT {
                    table_class,
                    destination,
                    code_lengths,
                    values,
                })
            },
        )
}

static UNZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

struct QuantizationTable([u16; 64]);

impl QuantizationTable {
    fn new(dqt: &DQT) -> Self {
        let mut quantization_table = [0u16; 64];

        for j in 0..64 {
            quantization_table[usize::from(UNZIGZAG[j])] = dqt.table[j];
        }

        Self(quantization_table)
    }
}

struct DQT {
    identifier: u8,
    table: [u16; 64],
}

impl Default for DQT {
    fn default() -> Self {
        DQT {
            identifier: u8::max_value(),
            table: [0; 64],
        }
    }
}

impl<A> Extend<A> for DQT
where
    A: Into<u16>,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = A>,
    {
        for (i, from) in (0..64).zip(iter) {
            self.table[i] = from.into();
        }
    }
}

#[derive(Copy, Clone)]
enum DQTPrecision {
    Bit8 = 0,
    Bit16 = 1,
}

impl DQTPrecision {
    fn new(b: u8) -> Option<Self> {
        Some(match b {
            0 => DQTPrecision::Bit8,
            1 => DQTPrecision::Bit16,
            _ => return None,
        })
    }
}

fn dqt<'a, I>() -> impl Parser<Output = DQT, Input = I> + 'a
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    split_4_bit()
        .and_then(|(precision, identifier)| -> Result<_, StreamErrorFor<I>> {
            Ok((
                DQTPrecision::new(precision).ok_or_else(|| {
                    StreamErrorFor::<I>::message_static_message("Unexpected DQT precision")
                })?,
                identifier,
            ))
        })
        .then_partial(|&mut (precision, identifier)| match precision {
            DQTPrecision::Bit8 => count_min_max(64, 64, any())
                .map(move |mut dqt: DQT| {
                    dqt.identifier = identifier;
                    dqt
                })
                .left(),
            DQTPrecision::Bit16 => count_min_max(64, 64, be_u16())
                .map(move |mut dqt: DQT| {
                    dqt.identifier = identifier;
                    dqt
                })
                .right(),
        })
        .message("DQT")
}

fn dri<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

#[derive(Debug)]
struct Scan {
    headers: Vec<ScanHeader>,
    start_of_selection: u8,
    end_of_selection: u8,
    high_approximation: u8,
    low_approximation: u8,
}

#[derive(Debug)]
struct ScanHeader {
    component_index: u8,
    dc_table_selector: u8,
    ac_table_selector: u8,
}

parser! {
fn sos['a, 'f, I](frame: &'f Frame)(I) -> Scan
where [
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
]
{
    let frame = *frame;
    let mut max_index: Option<u8> = None;
    (
        any().then_partial(move |&mut image_components| {
            let image_components = usize::from(image_components);
            count_min_max::<Vec<_>, _>(
                image_components,
                image_components,
                (any(), split_4_bit()).and_then(
                    move |(component_identifier, (dc_table_selector, ac_table_selector))| -> Result<_, StreamErrorFor<I>> {
                        debug_assert!(frame.components.len() <= 256);
                        let component_index = frame
                            .components
                            .iter()
                            .position(|c| c.component_identifier == component_identifier)
                            .ok_or_else(|| {
                                StreamErrorFor::<I>::message_static_message(
                                    "Component does not exist in frame",
                                )
                            })? as u8;

                        if let Some(max_index) = max_index {
                            use std::cmp::Ordering;
                            match component_index.cmp(&max_index) {
                                Ordering::Less =>
                                    return Err(StreamErrorFor::<I>::message_static_message(
                                        "Component index is smaller than the previous indicies"
                                    )),
                                Ordering::Equal =>
                                    return Err(StreamErrorFor::<I>::message_static_message(
                                        "Component index is is not unique"
                                    )),
                                Ordering::Greater => (),
                            }
                        }
                        max_index = Some(component_index);

                        Ok(ScanHeader {
                            component_index,
                            dc_table_selector,
                            ac_table_selector,
                        })
                    },
                ),
            )
        }),
        any(),
        any(),
        split_4_bit(),
    )
        .map(
            |(
                headers,
                start_of_selection,
                end_of_selection,
                (high_approximation, low_approximation),
            )| Scan {
                headers,
                start_of_selection,
                end_of_selection,
                high_approximation,
                low_approximation,
            },
        )
}
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum AdobeColorTransform {
    Unknown = 0,
    YCbCr = 1,
    YCCK = 2,
}

fn app_adobe<'a, I>() -> impl Parser<Output = AdobeColorTransform, Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    range(&b"Adobe\0"[..]).skip(take(5)).with(
        satisfy_map(|b| {
            Some(match b {
                0 => AdobeColorTransform::Unknown,
                1 => AdobeColorTransform::YCbCr,
                2 => AdobeColorTransform::YCCK,
                _ => return None,
            })
        })
        .expected("Adobe color transform"),
    )
}

#[derive(Default)]
pub struct Decoder {
    pub frame: Option<Frame>,
    scan: Option<Scan>,
    ac_huffman_tables: [Option<huffman::Table>; 16],
    dc_huffman_tables: [Option<huffman::Table>; 16],
    quantization_tables: [Option<QuantizationTable>; 4],
    planes: Vec<Vec<u8>>,
    color_transform: Option<AdobeColorTransform>,
}

impl Decoder {
    pub fn decode(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        {
            let decoder = std::cell::RefCell::new(&mut *self);
            let decoder = &decoder;
            let mut parser = many1::<Vec<_>, _>(segment().then_partial(|&mut segment| {
                parser(move |input| {
                    let x = decoder
                        .borrow_mut()
                        .do_segment::<easy::Stream<&[u8]>>(segment, input)
                        .map_err(|err| Consumed::Consumed(err.into()))?;
                    Ok((x, Consumed::Consumed(())))
                })
            }))
            .skip(eof());
            parser.easy_parse(input).map_err(|err| {
                err.map_position(|pos| pos.translate_position(input))
                    .map_token(|token| format!("0x{:X}", token))
                    .map_range(|range| format!("{:?}", range))
            })?;
        }

        let is_jfif = false; // TODO

        let frame = self.frame.as_ref().unwrap(); // FIXME

        let height = usize::from(frame.lines);
        let width = usize::from(frame.samples_per_line);

        let data = &self.planes;

        let color_convert_func = color_conversion::choose_color_convert_func(
            frame.components.len(),
            is_jfif,
            self.color_transform,
        )?;
        let upsampler =
            upsampler::Upsampler::new(&frame.components, frame.samples_per_line, frame.lines)?;
        let line_size = width * frame.components.len();
        let mut image = vec![0u8; line_size * height];

        for (row, line) in image.chunks_mut(line_size).enumerate() {
            upsampler.upsample_and_interleave_row(data, row, width, line);
            color_convert_func(line, width);
        }
        Ok(image)
    }

    fn decode_scan<'a>(
        &'a self,
        input: &mut huffman::Biterator,
    ) -> Result<impl Iterator<Item = Vec<u8>> + 'a, &'static str> {
        let frame = self.frame.as_ref().unwrap();
        let scan = self.scan.as_ref().unwrap();

        if scan.start_of_selection == 0
            && scan.headers.iter().any(|header| {
                self.dc_huffman_tables[usize::from(header.dc_table_selector)].is_none()
            })
        {
            return Err("scan uses unset dc huffman table");
        }

        if scan.end_of_selection > 1
            && scan.headers.iter().any(|header| {
                self.ac_huffman_tables[usize::from(header.ac_table_selector)].is_none()
            })
        {
            return Err("scan uses unset ac huffman table");
        }

        let mut mcu_row_coefficients: Vec<_> = frame
            .components
            .iter()
            .map(|component| {
                vec![
                    0i16;
                    usize::from(component.block_size.width)
                        * usize::from(component.vertical_sampling_factor)
                        * 64
                ]
            })
            .collect();
        let mut dc_predictors = [0i16; 4];
        let mut eob_run = 0;
        let is_interleaved = frame.components.len() > 1;

        let mut results: [Vec<_>; 4] = Default::default();
        for (component, result) in frame.components.iter().zip(&mut results[..]) {
            let size = usize::from(component.block_size.width)
                * usize::from(component.block_size.height)
                * 64;
            result.resize(size, 0u8);
        }

        let mut offsets = [0; 256];

        for mcu_y in 0..frame.mcu_size.height {
            for mcu_x in 0..frame.mcu_size.width {
                for (((component, dc_predictor), scan_header), mcu_row_coefficient) in frame
                    .components
                    .iter()
                    .zip(&mut dc_predictors)
                    .zip(&scan.headers)
                    .zip(&mut mcu_row_coefficients)
                {
                    let blocks_per_mcu = u16::from(component.horizontal_sampling_factor)
                        * u16::from(component.vertical_sampling_factor);
                    for j in 0..u16::from(blocks_per_mcu) {
                        let (block_x, block_y);

                        if is_interleaved {
                            block_x = mcu_x * u16::from(component.horizontal_sampling_factor)
                                + j % u16::from(component.horizontal_sampling_factor);
                            block_y = mcu_y * u16::from(component.vertical_sampling_factor)
                                + j / u16::from(component.horizontal_sampling_factor);
                        } else {
                            let blocks_per_row = usize::from(component.block_size.width);
                            let block_num = usize::from(
                                (mcu_y * frame.mcu_size.width + mcu_x * blocks_per_mcu) + j,
                            );

                            block_x = (block_num % blocks_per_row) as u16;
                            block_y = (block_num / blocks_per_row) as u16;

                            if block_x * 8 >= component.size.width
                                || block_y * 8 >= component.size.height
                            {
                                continue;
                            }
                        }

                        let block_offset = (usize::from(block_y)
                            * usize::from(component.block_size.width)
                            + usize::from(block_x))
                            * 64;
                        let mcu_row_offset = usize::from(mcu_y)
                            * usize::from(component.block_size.width)
                            * usize::from(component.vertical_sampling_factor)
                            * 64;
                        let coefficients = &mut mcu_row_coefficient
                            [block_offset - mcu_row_offset..block_offset - mcu_row_offset + 64];

                        let dc_table = self.dc_huffman_tables
                            [usize::from(scan_header.dc_table_selector)]
                        .as_ref()
                        .unwrap_or_else(|| panic!("Missing DC table")); // TODO un-unwrap
                        let ac_table = self.ac_huffman_tables
                            [usize::from(scan_header.ac_table_selector)]
                        .as_ref()
                        .unwrap_or_else(|| panic!("Missing AC table")); // TODO un-unwrap

                        if scan.high_approximation == 0 {
                            self.decode_block(
                                coefficients,
                                scan,
                                &dc_table,
                                &ac_table,
                                input,
                                &mut eob_run,
                                dc_predictor,
                            )?;
                        } else {
                            unimplemented!()
                        }
                    }
                }
            }

            for (component_index, component) in frame.components.iter().enumerate() {
                let row_coefficients = if frame.coding_process() == CodingProcess::Progressive {
                    unimplemented!()
                } else {
                    &mut mcu_row_coefficients[component_index]
                };

                // Convert coefficients from a MCU row to samples.
                let data = row_coefficients;
                let offset = offsets[component_index];

                let quantization_table = self.quantization_tables
                    [usize::from(component.quantization_table_destination_selector)]
                .as_ref()
                .unwrap_or_else(|| {
                    panic!(
                        "Missing quantization table {}",
                        component.quantization_table_destination_selector
                    )
                });

                let block_count = usize::from(component.block_size.width)
                    * usize::from(component.vertical_sampling_factor);
                let line_stride = usize::from(component.block_size.width) * 8;

                assert_eq!(data.len(), block_count * 64);

                let component_result = &mut results[component_index];
                for i in 0..block_count {
                    let x = (i % usize::from(component.block_size.width)) * 8;
                    let y = (i / usize::from(component.block_size.width)) * 8;
                    idct::dequantize_and_idct_block(
                        &data[i * 64..(i + 1) * 64],
                        &quantization_table.0,
                        line_stride,
                        &mut component_result[offset + y * line_stride + x..],
                    );
                }

                offsets[component_index] += data.len();
            }
        }
        Ok(scan.headers.iter().map(move |header| {
            mem::replace(
                &mut results[usize::from(header.component_index)],
                Default::default(),
            )
        }))
    }

    fn decode_block(
        &self,
        coefficients: &mut [i16],
        scan: &Scan,
        dc_table: &huffman::Table,
        ac_table: &huffman::Table,
        input: &mut huffman::Biterator,
        eob_run: &mut u16,
        dc_predictor: &mut i16,
    ) -> Result<(), &'static str> {
        if scan.start_of_selection == 0 {
            let value = dc_table
                .decode(input)
                .ok_or_else(|| "Unable to huffman decode")?;
            let diff = if value == 0 {
                0
            } else {
                if value > 11 {
                    return Err("Invalid DC difference magnitude category");
                }
                input
                    .receive_extend(value)
                    .ok_or_else(|| "Out of input in receive extend")?
            };

            // Malicious JPEG files can cause this add to overflow, therefore we use wrapping_add.
            // One example of such a file is tests/crashtest/images/dc-predictor-overflow.jpg
            *dc_predictor = dc_predictor.wrapping_add(diff);
            coefficients[0] = *dc_predictor << scan.low_approximation;
        }

        let mut index = scan.start_of_selection.max(1);

        if index <= scan.end_of_selection && *eob_run > 0 {
            *eob_run -= 1;
            return Ok(());
        }

        // Section F.1.2.2.1
        while index <= scan.end_of_selection {
            if let Some((value, run)) = ac_table
                .decode_fast_ac(input)
                .map_err(|()| "Unable to huffman decode")?
            {
                index += run;

                if index > scan.end_of_selection {
                    break;
                }

                coefficients[usize::from(UNZIGZAG[usize::from(index)])] =
                    value << scan.low_approximation;
                index += 1;
            } else {
                let byte = ac_table
                    .decode(input)
                    .ok_or_else(|| "Unable to huffman decode")?;
                let r = byte >> 4;
                let s = byte & 0x0f;

                if s == 0 {
                    match r {
                        15 => index += 16, // Run length of 16 zero coefficients.
                        _ => {
                            *eob_run = (1 << r) - 1;

                            if r > 0 {
                                *eob_run += input.next_bits(r).ok_or_else(|| "out of bits")?;
                            }

                            break;
                        }
                    }
                } else {
                    index += r;

                    if index > scan.end_of_selection {
                        break;
                    }

                    coefficients[usize::from(UNZIGZAG[usize::from(index)])] = input
                        .receive_extend(s)
                        .ok_or_else(|| "Invalid receive_extend")?
                        << scan.low_approximation;
                    index += 1;
                }
            }
        }

        Ok(())
    }

    fn do_segment<'a, I>(&mut self, segment: Segment<'a>, input: &mut I) -> Result<(), I::Error>
    where
        I: FullRangeStream<Item = u8, Range = &'a [u8]> + From<&'a [u8]> + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
    {
        log::trace!("Segment {:?}", segment);
        match segment.marker {
            Marker::SOI | Marker::RST(_) | Marker::COM | Marker::EOI => Ok(()),
            Marker::SOF(i) => {
                self.frame = Some(sof(i).parse(I::from(segment.data))?.0);
                Ok(())
            }
            Marker::DHT => {
                skip_many1(huffman_table().map(|dht| {
                    let table =
                        huffman::Table::new(dht.code_lengths, dht.values, dht.table_class).unwrap(); // FIXME
                    match dht.table_class {
                        huffman::TableClass::AC => {
                            self.ac_huffman_tables[usize::from(dht.destination)] = Some(table)
                        }
                        huffman::TableClass::DC => {
                            self.dc_huffman_tables[usize::from(dht.destination)] = Some(table)
                        }
                    }
                }))
                .parse(I::from(segment.data))?
                .0;
                Ok(())
            }
            Marker::DQT => {
                skip_many1(dqt().map(|dqt| {
                    self.quantization_tables[usize::from(dqt.identifier)] =
                        Some(QuantizationTable::new(&dqt));
                }))
                .parse(I::from(segment.data))?;
                Ok(())
            }
            Marker::DRI => dri().parse(I::from(segment.data)).map(|_| ()),
            Marker::SOS => {
                let header_input = I::from(segment.data);
                let frame = self.frame.as_ref().ok_or_else(|| {
                    I::Error::from_error(
                        input.position(),
                        StreamErrorFor::<I>::message_static_message("Found SOS before SOF"),
                    )
                })?;
                let (scan, _rest) = sos(frame).parse(header_input)?;
                self.scan = Some(scan);

                let mut biterator = huffman::Biterator::new(input.range());
                let planes = self
                    .decode_scan(&mut biterator)
                    .map_err(|msg| {
                        let input = I::from(segment.data);
                        I::Error::from_error(
                            input.position(),
                            StreamErrorFor::<I>::message_static_message(msg),
                        )
                    })?
                    .collect();
                self.planes = planes;
                *input = I::from(biterator.into_inner());

                Ok(())
            }
            Marker::APP_ADOBE => {
                self.color_transform = Some(app_adobe().parse(I::from(segment.data))?.0);
                Ok(())
            }
            Marker::APP(_) => Ok(()),
        }
    }
}

pub fn decode(input: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = Decoder::default();

    decoder.decode(input)
}
