use std::{fmt, mem};

use arrayvec::ArrayVec;

use derive_more::{Display, From};

use combine::{
    dispatch, easy,
    error::{Consumed, ParseError, StreamError},
    parser,
    parser::{
        byte::num::be_u16,
        item::{any, eof, satisfy_map, value},
        range::{range, take},
        repeat::{count_min_max, iterate, skip_many1},
        ParseMode,
    },
    stream::{
        user_state::StateStream, FullRangeStream, PointerOffset, Positioned, ResetStream,
        StreamErrorFor,
    },
    ParseResult, Parser, RangeStreamOnce, Stream, StreamOnce,
};

use {
    biterator::Biterator,
    marker::{marker, Marker},
};

mod biterator;
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
    Format(String),

    #[display(fmt = "{}", _0)]
    Message(&'static str),
}

type Result<T, E = Error> = std::result::Result<T, E>;

const MAX_COMPONENTS: usize = 4;

fn split_4_bit<'a, I>() -> impl Parser<I, Output = (u8, u8)>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b >> 4, b & 0x0F))
}

fn segment<'a, 's, I, P, O>(mut parser: P) -> impl Parser<DecoderStream<'s, I>, Output = O> + 'a
where
    P: Parser<DecoderStream<'s, I>, Output = O> + 'a,
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + From<&'a [u8]> + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
    's: 'a,
{
    be_u16()
        .then_partial(|&mut len| take(usize::from(len - 2))) // Check len >= 2
        .map_input(
            move |data, self_: &mut DecoderStream<'s, I>| -> Result<_, I::Error> {
                let before = mem::replace(self_.stream.as_inner_mut(), I::from(data));

                let result = parser.parse_with_state(self_, &mut Default::default());

                *self_.stream.as_inner_mut() = before;

                result
            },
        )
        .flat_map(move |data| -> Result<_, I::Error> { data })
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
    fn is_baseline(&self) -> bool {
        self.marker_index == 0
    }

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

fn sof<'a, I>(marker_index: u8) -> impl Parser<I, Output = Frame>
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

fn huffman_table<'a, I>() -> impl Parser<I, Output = DHT<'a>> + 'a
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

fn dqt<'a, I>() -> impl Parser<I, Output = DQT> + 'a
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

fn dri<'a, I>() -> impl Parser<I, Output = u16>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    be_u16()
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

fn sos_segment<'a, 's, I>() -> impl Parser<DecoderStream<'s, I>, Output = Scan> + 'a
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
    's: 'a,
{
    let mut max_index: Option<u8> = None;
    (
        any().then_partial(move |&mut image_components| {
            let image_components = usize::from(image_components);
            count_min_max::<Vec<_>, _, _>(
                image_components,
                image_components,
                (any(), split_4_bit())
                    .map_input(
                        move |(component_identifier, (dc_table_selector, ac_table_selector)),
                              self_: &mut DecoderStream<'s, I>|
                              -> Result<_, StreamErrorFor<I>> {
                            let frame = self_.state.frame.as_ref().ok_or_else(|| {
                                StreamErrorFor::<I>::message_static_message("Found SOS before SOF")
                            })?;
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
                                    Ordering::Less => {
                                        return Err(StreamErrorFor::<I>::message_static_message(
                                            "Component index is smaller than the previous indicies",
                                        ))
                                    }
                                    Ordering::Equal => {
                                        return Err(StreamErrorFor::<I>::message_static_message(
                                            "Component index is is not unique",
                                        ))
                                    }
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
                    )
                    .and_then(|result| result),
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

struct InputConverter<P> {
    parser: P,
}
impl<'s, 'a, Input, P, O, S> Parser<DecoderStream<'s, Input>> for InputConverter<P>
where
    Input: FullRangeStream<Item = u8, Range = &'a [u8]>,
    Input::Error: ParseError<Input::Item, Input::Range, Input::Position>,
    Input::Position: Default + fmt::Display,
    P: Parser<BiteratorStream<'s, Input>, Output = O, PartialState = S>,
    S: Default,
{
    type Output = O;
    type PartialState = S;

    combine::parse_mode!(DecoderStream<'s, Input>);

    fn parse_mode_impl<M>(
        &mut self,
        mode: M,
        input: &mut DecoderStream<'s, Input>,
        state: &mut Self::PartialState,
    ) -> ParseResult<Self::Output, Input::Error>
    where
        M: ParseMode,
    {
        self.parser
            .parse_mode(mode, &mut input.0, state)
            .map_err(|err| {
                Input::Error::from_error(
                    input.position(),
                    StreamErrorFor::<Input>::message_message(err), // FIXME
                )
            })
    }
}

fn sos<'a, 's, I>() -> impl Parser<DecoderStream<'s, I>, Output = ()> + 'a
where
    I: FullRangeStream<
            Item = u8,
            Range = &'a [u8],
            Error = easy::Errors<u8, &'a [u8], PointerOffset<[u8]>>,
        > + From<&'a [u8]>
        + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
    I::Position: Default + fmt::Display,
    's: 'a,
{
    segment(sos_segment())
        .map_input(move |scan, input: &mut DecoderStream<'s, I>| {
            input.state.scan = Some(scan);
            let scan = input.0.state.scan.as_ref().unwrap();

            // FIXME Propagate error
            let frame = input.0.state.frame.as_ref().ok_or_else(|| {
                I::Error::from_error(
                    input.0.position(),
                    StreamErrorFor::<I>::message_static_message("Found SOS before SOF"),
                )
            })?;
            if frame.coding_process() == CodingProcess::Progressive {
                unimplemented!();
            }

            if scan.low_approximation == 0 {
                for i in scan.headers.iter().map(|header| header.component_index) {
                    for j in scan.start_of_selection..=scan.end_of_selection {
                        input.0.state.coefficients_finished[usize::from(i)] |= 1 << j;
                    }
                }
            }

            let is_final_scan = scan.headers.iter().all(|header| {
                input.0.state.coefficients_finished[usize::from(header.component_index)] == !0
            });
            Ok(is_final_scan)
        })
        .flat_map(move |data| -> Result<_, I::Error> { data })
        .then(move |is_final_scan: bool| InputConverter {
            parser: Decoder::decode_scan(is_final_scan)
                .map_input(|iter, input| input.state.planes = iter),
        })
}

#[repr(transparent)]
struct DecoderStream<'s, I>(BiteratorStream<'s, I>);

fn decoder_stream<'s, 't, I>(s: &'t mut BiteratorStream<'s, I>) -> &'t mut DecoderStream<'s, I> {
    // SAFETY repr(transparent) is defined on `DecoderStream`
    unsafe { mem::transmute(s) }
}

impl<'s, I> std::ops::Deref for DecoderStream<'s, I> {
    type Target = BiteratorStream<'s, I>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'s, I> std::ops::DerefMut for DecoderStream<'s, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'s, I> StreamOnce for DecoderStream<'s, I>
where
    I: StreamOnce,
{
    type Item = I::Item;
    type Range = I::Range;
    type Position = I::Position;
    type Error = I::Error;

    #[inline]
    fn uncons(&mut self) -> Result<I::Item, StreamErrorFor<Self>> {
        self.0.stream.as_inner_mut().uncons()
    }
}

impl<'s, I> ResetStream for DecoderStream<'s, I>
where
    I: ResetStream,
{
    type Checkpoint = I::Checkpoint;

    fn checkpoint(&self) -> Self::Checkpoint {
        self.0.stream.as_inner().checkpoint()
    }

    fn reset(&mut self, checkpoint: Self::Checkpoint) -> Result<(), Self::Error> {
        self.0.stream.as_inner_mut().reset(checkpoint)
    }
}

impl<'s, I> RangeStreamOnce for DecoderStream<'s, I>
where
    I: RangeStreamOnce,
{
    fn uncons_range(&mut self, size: usize) -> Result<Self::Range, StreamErrorFor<Self>> {
        self.stream.as_inner_mut().uncons_range(size)
    }

    fn uncons_while<F>(&mut self, f: F) -> Result<Self::Range, StreamErrorFor<Self>>
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.stream.as_inner_mut().uncons_while(f)
    }

    fn uncons_while1<F>(&mut self, f: F) -> ParseResult<Self::Range, StreamErrorFor<Self>>
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.stream.as_inner_mut().uncons_while1(f)
    }

    fn distance(&self, end: &Self::Checkpoint) -> usize {
        self.stream.as_inner().distance(end)
    }
}

impl<'s, I> FullRangeStream for DecoderStream<'s, I>
where
    I: FullRangeStream,
{
    fn range(&self) -> Self::Range {
        self.stream.as_inner().range()
    }
}

impl<'s, I> Positioned for DecoderStream<'s, I>
where
    I: Positioned,
{
    fn position(&self) -> Self::Position {
        self.0.stream.as_inner().position()
    }
}

type BiteratorStream<'s, I> = StateStream<Biterator<I>, &'s mut Decoder>;

#[derive(PartialEq, Eq, Clone, Copy)]
enum AdobeColorTransform {
    Unknown = 0,
    YCbCr = 1,
    YCCK = 2,
}

fn app_adobe<'a, I>() -> impl Parser<I, Output = AdobeColorTransform> + 'a
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
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
    restart_interval: u16,
    coefficients_finished: [u64; MAX_COMPONENTS],

    scan_state: ScanState,
}

struct ScanState {
    mcu_row_coefficients: Vec<Vec<i16>>,
    dummy_block: [i16; 64],
    dc_predictors: [i16; 4],
    eob_run: u16,
    expected_rst_num: u8,
    mcus_left_until_restart: u16,
    is_interleaved: bool,
    results: [Vec<u8>; 4],
    offsets: [usize; 256],
}

impl Default for ScanState {
    fn default() -> Self {
        ScanState {
            mcu_row_coefficients: vec![],
            dummy_block: [0; 64],
            dc_predictors: [0; 4],
            eob_run: 0,
            expected_rst_num: 0,
            mcus_left_until_restart: 0,
            is_interleaved: false,
            results: Default::default(),
            offsets: [0; 256],
        }
    }
}

impl Decoder {
    pub fn decode(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        {
            let mut parser =
                skip_many1(marker().then_partial(move |&mut marker| Self::do_segment(marker)))
                    .skip(eof());
            parser
                .parse(DecoderStream(StateStream {
                    stream: biterator::Biterator::new(easy::Stream(input)),
                    state: self,
                }))
                .map_err(|err| {
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

    fn decode_scan<'s, 'a, I>(
        produce_data: bool,
    ) -> impl Parser<BiteratorStream<'s, I>, Output = Vec<Vec<u8>>>
    where
        I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
        I::Position: Default + fmt::Display,
        's: 'a,
    {
        parser(move |input: &mut BiteratorStream<'s, I>| {
            let frame = input.state.frame.as_ref().unwrap();
            let scan = input.state.scan.as_ref().unwrap();

            if scan.start_of_selection == 0
                && scan.headers.iter().any(|header| {
                    input.state.dc_huffman_tables[usize::from(header.dc_table_selector)].is_none()
                })
            {
                return Err(Consumed::Consumed(
                    <BiteratorStream<I> as StreamOnce>::Error::from_error(
                        input.position(),
                        StreamErrorFor::<BiteratorStream<I>>::message_static_message(
                            "scan uses unset dc huffman table",
                        ),
                    )
                    .into(),
                ));
            }

            if scan.end_of_selection > 1
                && scan.headers.iter().any(|header| {
                    input.state.ac_huffman_tables[usize::from(header.ac_table_selector)].is_none()
                })
            {
                return Err(Consumed::Consumed(
                    <BiteratorStream<I> as StreamOnce>::Error::from_error(
                        input.position(),
                        StreamErrorFor::<BiteratorStream<I>>::message_static_message(
                            "scan uses unset ac huffman table",
                        ),
                    )
                    .into(),
                ));
            }

            let mcu_row_coefficients: Vec<_> =
                if produce_data && frame.coding_process() != CodingProcess::Progressive {
                    frame
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
                        .collect()
                } else {
                    Vec::new()
                };

            input.state.scan_state = ScanState {
                dummy_block: [0i16; 64],
                dc_predictors: [0i16; 4],
                eob_run: 0,
                expected_rst_num: 0,
                mcus_left_until_restart: input.state.restart_interval,
                is_interleaved: frame.components.len() > 1,
                mcu_row_coefficients,
                results: Default::default(),
                offsets: [0; 256],
            };

            if produce_data {
                for (component, result) in frame
                    .components
                    .iter()
                    .zip(&mut input.state.scan_state.results[..])
                {
                    let size = usize::from(component.block_size.width)
                        * usize::from(component.block_size.height)
                        * 64;
                    result.resize(size, 0u8);
                }
            }

            Ok(((), Consumed::Empty(())))
        })
        .map_input(move |_, input: &mut BiteratorStream<'s, I>| {
            Self::iteration_decode_scanner(produce_data, input)
        })
        .then(|parser| parser)
        .map_input(move |_, input: &mut BiteratorStream<'s, I>| {
            let results = &mut input.state.scan_state.results;
            input
                .state
                .scan
                .as_ref()
                .unwrap()
                .headers
                .iter()
                .map(move |header| {
                    mem::replace(
                        &mut results[usize::from(header.component_index)],
                        Default::default(),
                    )
                })
                .collect()
        })
    }

    fn iteration_decode_scanner<'a, 's, I>(
        produce_data: bool,
        input: &mut BiteratorStream<'s, I>,
    ) -> impl Parser<BiteratorStream<'s, I>, Output = ()>
    where
        I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
        I::Position: Default + fmt::Display,
        's: 'a,
    {
        let frame = input.state.frame.as_ref().unwrap();

        let mcu_width = frame.mcu_size.width;
        let components_len = frame.components.len();
        iterate::<(), _, _, _, _>(0..frame.mcu_size.height, move |&mcu_y, _| {
            iterate::<(), _, _, _, _>(0..mcu_width, move |&mcu_x, _| {
                (
                    iterate::<(), _, _, _, _>(
                        0..components_len,
                        move |&i, input: &mut BiteratorStream<'s, I>| {
                            let frame = input.state.frame.as_ref().unwrap();

                            let component = &frame.components[i];
                            let blocks_per_mcu = u16::from(component.horizontal_sampling_factor)
                                * u16::from(component.vertical_sampling_factor);
                            iterate::<(), _, _, _, _>(0..u16::from(blocks_per_mcu), move |&j, _| {
                                Self::decode_block_parser(
                                    mcu_x,
                                    mcu_y,
                                    i,
                                    j,
                                    blocks_per_mcu,
                                    produce_data,
                                )
                            })
                        },
                    ),
                    Self::restart_parser(mcu_x, mcu_y),
                )
                    .map(|_| ())
            })
            .map_input(move |_, input: &mut BiteratorStream<'s, I>| {
                if produce_data {
                    let frame = input.state.frame.as_ref().unwrap();

                    for (component_index, component) in frame.components.iter().enumerate() {
                        let coefficients_per_mcu_row = component.block_size.width as usize
                            * component.vertical_sampling_factor as usize
                            * 64;
                        let row_coefficients = if frame.coding_process()
                            == CodingProcess::Progressive
                        {
                            unimplemented!()
                        } else {
                            mem::replace(
                                &mut input.state.scan_state.mcu_row_coefficients[component_index],
                                vec![0; coefficients_per_mcu_row],
                            )
                        };

                        // Convert coefficients from a MCU row to samples.
                        let data = row_coefficients;
                        let offset = input.state.scan_state.offsets[component_index];

                        let quantization_table = input.state.quantization_tables
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

                        let component_result = &mut input.state.scan_state.results[component_index];
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

                        input.state.scan_state.offsets[component_index] += data.len();
                    }
                }
            })
        })
    }

    fn restart_parser<'s, 'a, I>(
        mcu_x: u16,
        mcu_y: u16,
    ) -> impl Parser<BiteratorStream<'s, I>, Output = ()>
    where
        I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
        I::Position: Default + fmt::Display,
        's: 'a,
    {
        parser(move |input: &mut BiteratorStream<'s, I>| {
            if input.state.restart_interval > 0 {
                let frame = input.state.frame.as_ref().unwrap();
                let is_last_mcu =
                    mcu_x == frame.mcu_size.width - 1 && mcu_y == frame.mcu_size.height - 1;
                input.state.scan_state.mcus_left_until_restart -= 1;

                if input.state.scan_state.mcus_left_until_restart == 0 && !is_last_mcu {
                    match marker()
                        .parse_stream(decoder_stream(input))
                        .into_result()
                        .map(|(marker, _)| marker)
                    {
                        Ok(Marker::RST(n)) => {
                            if n != input.state.scan_state.expected_rst_num {
                                return Err(Consumed::Consumed(
                                    <BiteratorStream<I> as StreamOnce>::Error::from_error(
                                        input.position(),
                                        StreamErrorFor::<BiteratorStream<I>>::message_message(
                                            format_args!(
                                                "found RST{} where RST{} was expected",
                                                n, input.state.scan_state.expected_rst_num
                                            ),
                                        ),
                                    )
                                    .into(),
                                ));
                            }

                            input.stream.reset();

                            // Section F.2.1.3.1
                            input.state.scan_state.dc_predictors = [0i16; MAX_COMPONENTS];
                            // Section G.1.2.2
                            input.state.scan_state.eob_run = 0;

                            input.state.scan_state.expected_rst_num =
                                (input.state.scan_state.expected_rst_num + 1) % 8;
                            input.state.scan_state.mcus_left_until_restart =
                                input.state.restart_interval;
                        }
                        Ok(marker) => {
                            return Err(Consumed::Consumed(
                                                <BiteratorStream<I> as StreamOnce>::Error::from_error(
                                                    input.position(),
                                                    StreamErrorFor::<BiteratorStream<I>>::message_message(
                                                                        format_args!(
                                                                            "found marker {:?} inside scan where RST({}) was expected",
                                                                            marker,
                                                                            input.state.scan_state.expected_rst_num
                                                                        )
                                                                    )
                                                        )
                                                .into()
                                                    )
                                            );
                        }
                        Err(_) => {
                            return Err(Consumed::Consumed(
                                <BiteratorStream<I> as StreamOnce>::Error::from_error(
                                    input.position(),
                                    StreamErrorFor::<BiteratorStream<I>>::message_message(
                                        format_args!(
                                            "no marker found where RST{} was expected",
                                            input.state.scan_state.expected_rst_num
                                        ),
                                    ),
                                )
                                .into(),
                            ));
                        }
                    }
                }
            }
            Ok(((), Consumed::Empty(())))
        })
    }

    fn decode_block_parser<'s, 'a, I>(
        mcu_x: u16,
        mcu_y: u16,
        i: usize,
        j: u16,
        blocks_per_mcu: u16,
        produce_data: bool,
    ) -> impl Parser<BiteratorStream<'s, I>, Output = ()>
    where
        I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
        I::Position: Default + fmt::Display,
        's: 'a,
    {
        parser(move |input: &mut BiteratorStream<'s, I>| {
            let frame = input.state.frame.as_ref().unwrap();
            let scan = input.state.scan.as_ref().unwrap();

            let component = &frame.components[i];
            let dc_predictor = &mut input.state.scan_state.dc_predictors[i];
            let scan_header = &scan.headers[i];

            let (block_x, block_y);

            if input.state.scan_state.is_interleaved {
                block_x = mcu_x * u16::from(component.horizontal_sampling_factor)
                    + j % u16::from(component.horizontal_sampling_factor);
                block_y = mcu_y * u16::from(component.vertical_sampling_factor)
                    + j / u16::from(component.horizontal_sampling_factor);
            } else {
                let blocks_per_row = usize::from(component.block_size.width);
                let block_num =
                    usize::from((mcu_y * frame.mcu_size.width + mcu_x * blocks_per_mcu) + j);

                block_x = (block_num % blocks_per_row) as u16;
                block_y = (block_num / blocks_per_row) as u16;

                if block_x * 8 >= component.size.width || block_y * 8 >= component.size.height {
                    return Ok(((), Consumed::Empty(())));
                }
            }

            let block_offset = (usize::from(block_y) * usize::from(component.block_size.width)
                + usize::from(block_x))
                * 64;
            let mcu_row_offset = usize::from(mcu_y)
                * usize::from(component.block_size.width)
                * usize::from(component.vertical_sampling_factor)
                * 64;
            let coefficients = if produce_data {
                &mut input.state.scan_state.mcu_row_coefficients[i]
                    [block_offset - mcu_row_offset..block_offset - mcu_row_offset + 64]
            } else {
                &mut input.state.scan_state.dummy_block[..]
            };

            let dc_table = input.state.dc_huffman_tables
                [usize::from(scan_header.dc_table_selector)]
            .as_ref()
            .unwrap_or_else(|| panic!("Missing DC table")); // TODO un-unwrap
            let ac_table = input.state.ac_huffman_tables
                [usize::from(scan_header.ac_table_selector)]
            .as_ref()
            .unwrap_or_else(|| panic!("Missing AC table")); // TODO un-unwrap

            if scan.high_approximation == 0 {
                Self::decode_block(
                    coefficients,
                    scan,
                    &dc_table,
                    &ac_table,
                    &mut input.stream,
                    &mut input.state.scan_state.eob_run,
                    dc_predictor,
                )
                .map(|()| ((), Consumed::Consumed(())))
                .map_err(|msg| {
                    Consumed::Consumed(
                        <BiteratorStream<I> as StreamOnce>::Error::from_error(
                            input.position(),
                            StreamErrorFor::<BiteratorStream<I>>::message_static_message(msg),
                        )
                        .into(),
                    )
                })
            } else {
                unimplemented!()
            }
        })
    }

    fn decode_block<I>(
        coefficients: &mut [i16],
        scan: &Scan,
        dc_table: &huffman::Table,
        ac_table: &huffman::Table,
        input: &mut biterator::Biterator<I>,
        eob_run: &mut u16,
        dc_predictor: &mut i16,
    ) -> Result<(), &'static str>
    where
        I: Stream<Item = u8>,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
        I::Position: Default,
    {
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

    fn do_segment<'a, 's, I>(marker: Marker) -> impl Parser<DecoderStream<'s, I>, Output = ()> + 'a
    where
        I: FullRangeStream<
                Item = u8,
                Range = &'a [u8],
                Error = easy::Errors<u8, &'a [u8], PointerOffset<[u8]>>,
            > + From<&'a [u8]>
            + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
        I::Position: Default + fmt::Display,
        's: 'a,
    {
        log::trace!("Segment {:?}", marker);

        dispatch!(marker;
            Marker::SOI | Marker::RST(_) | Marker::COM | Marker::EOI => value(()),
            Marker::SOF(i) => segment(sof(i)).map_input(move |frame, self_: &mut DecoderStream<'s, I>| self_.state.frame = Some(frame)),
            Marker::DHT => {
                segment(skip_many1(huffman_table().map_input(move |dht, self_: &mut DecoderStream<'s, I>| {
                    let table =
                        huffman::Table::new(dht.code_lengths, dht.values, dht.table_class)
                            .unwrap(); // FIXME
                    match dht.table_class {
                        huffman::TableClass::AC => {
                            self_.state.ac_huffman_tables[usize::from(dht.destination)] = Some(table)
                        }
                        huffman::TableClass::DC => {
                            self_.state.dc_huffman_tables[usize::from(dht.destination)] = Some(table)
                        }
                    }
                })))
            },
            Marker::DQT => segment(skip_many1(dqt().map_input(move |dqt, self_: &mut DecoderStream<'s, I>| {
                self_.state.quantization_tables[usize::from(dqt.identifier)] =
                    Some(QuantizationTable::new(&dqt));
            }))),
            Marker::DRI => segment(dri()).map_input(move |r, self_: &mut DecoderStream<'s, I>| self_.state.restart_interval = r),
            Marker::SOS => sos(),
            Marker::APP_ADOBE => {
                segment(app_adobe())
                    .map_input(move |color_transform, self_: &mut DecoderStream<'s, I>| self_.state.color_transform = Some(color_transform))
            },
            Marker::APP(_) => segment(value(())),
        )
    }
}

pub fn decode(input: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = Decoder::default();

    decoder.decode(input)
}
