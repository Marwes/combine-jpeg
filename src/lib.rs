use combine::{
    easy,
    error::{ParseError, StreamError},
    parser,
    parser::{
        byte::{byte, num::be_u16, take_until_byte},
        item::{any, eof, satisfy_map, value},
        range::{take, take_while1},
        repeat::{count_min_max, many1, sep_by1, skip_many1},
    },
    stream::{FullRangeStream, StreamErrorFor},
    Parser,
};

mod huffman;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Marker {
    SOI,
    // SOF 0 : Baseline DCT
    // SOF 1 : Extended sequential DCT, Huffman coding
    // SOF 2 : Progressive DCT, Huffman coding
    // SOF 3 : Lossless (sequential), Huffman coding
    // SOF 9 : Extended sequential DCT, arithmetic coding
    // SOF 10 : Progressive DCT, arithmetic coding
    // SOF 11 : Lossless (sequential), arithmetic coding
    SOF(u8),
    DHT,
    DQT,
    DRI,
    SOS,
    RST(u8),
    APP(u8),
    COM(u8),
    EOI,
}

impl Marker {
    const APP_ADOBE: Marker = Marker::APP(14);
}

fn marker<'a, I>() -> impl Parser<Output = Marker, Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    #[derive(Clone)]
    pub struct Sink;

    impl Default for Sink {
        fn default() -> Self {
            Sink
        }
    }

    impl<A> Extend<A> for Sink {
        fn extend<T>(&mut self, iter: T)
        where
            T: IntoIterator<Item = A>,
        {
            for _ in iter {}
        }
    }

    sep_by1::<Sink, _, _>(
        (
            take_until_byte(0xFF),      // mozjpeg skips any non marker bytes (non 0xFF)
            take_while1(|b| b == 0xFF), // Extraenous 0xFF bytes are allowed
        ),
        byte(0x00).expected("stuffed zero"), // When we encounter a 0x00, we found a stuffed zero (FF/00) sequence so we search again
    )
    .with(
        satisfy_map(|b| {
            Some(match b {
                0xD8 => Marker::SOI,
                0xC0...0xC2 => Marker::SOF(b - 0xC0),
                0xC4 => Marker::DHT,
                0xDB => Marker::DQT,
                0xDD => Marker::DRI,
                0xDA => Marker::SOS,
                0xD0...0xD7 => Marker::RST(b - 0xD0),
                0xE0...0xEF => Marker::APP(b - 0xE0),
                0xD9 => Marker::EOI,
                _ => return None,
            })
        })
        .expected("marker"),
    )
}

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
        Marker::SOF(_) | Marker::DHT | Marker::DQT | Marker::SOS | Marker::APP(_) => be_u16() // TODO Check length >= 2
            .then(|quantization_table_len| take((quantization_table_len - 2).into()))
            .map(move |data| Segment { marker, data })
            .right(),
        _ => panic!("Unhandled marker {:?}", marker),
    }
    })
}

#[derive(Default, Copy, Clone)]
struct Dimensions {
    width: u16,
    height: u16,
}

struct Frame {
    precision: u8,
    lines: u16,
    samples_per_line: u16,
    mcu_size: Dimensions,
    components: [ComponentSpecification; 256],
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            precision: 0,
            lines: 0,
            samples_per_line: 0,
            mcu_size: Default::default(),
            components: [ComponentSpecification::default(); 256],
        }
    }
}

impl Extend<ComponentSpecification> for Frame {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = ComponentSpecification>,
    {
        for (to, from) in self.components.iter_mut().zip(iter) {
            *to = from;
        }
    }
}

impl Frame {
    fn component_width(&self, index: usize) -> u16 {
        self.mcu_size.width * u16::from(self.components[index].horizontal_sampling_factor)
    }
}

#[derive(Copy, Clone, Default)]
struct ComponentSpecification {
    component_identifier: u8,
    horizontal_sampling_factor: u8,
    vertical_sampling_factor: u8,
    quantization_table_destination_selector: u8,

    // Computed parts
    block_size: Dimensions,
    size: Dimensions,
}

fn sof0<'a, I>() -> impl Parser<Output = Frame, Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    sof2()
}

fn sof2<'a, I>() -> impl Parser<Output = Frame, Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    (any(), be_u16(), be_u16(), any()).then_partial(
        |&mut (precision, lines, samples_per_line, components_in_frame)| {
            let component = (any(), split_4_bit(), any()).map(
                |(
                    component_identifier,
                    (horizontal_sampling_factor, vertical_sampling_factor),
                    quantization_table_destination_selector,
                )| ComponentSpecification {
                    component_identifier,
                    horizontal_sampling_factor,
                    vertical_sampling_factor,
                    quantization_table_destination_selector,

                    // Filled in later
                    block_size: Default::default(),
                    size: Default::default(),
                },
            );
            count_min_max::<Frame, _>(
                usize::from(components_in_frame),
                usize::from(components_in_frame),
                component,
            )
            .map(move |mut frame| {
                frame.precision = precision;
                frame.lines = lines;
                frame.samples_per_line = samples_per_line;

                {
                    let h_max = f32::from(
                        frame
                            .components
                            .iter()
                            .map(|c| c.horizontal_sampling_factor)
                            .max()
                            .unwrap(),
                    );
                    let v_max = f32::from(
                        frame
                            .components
                            .iter()
                            .map(|c| c.vertical_sampling_factor)
                            .max()
                            .unwrap(),
                    );

                    let samples_per_line = f32::from(samples_per_line);
                    let lines = f32::from(lines);

                    frame.mcu_size = Dimensions {
                        width: (samples_per_line / (h_max * 8.0)).ceil() as u16,
                        height: (lines / (v_max * 8.0)).ceil() as u16,
                    };

                    for component in &mut frame.components[..] {
                        component.size.width = (f32::from(samples_per_line)
                            * (f32::from(component.horizontal_sampling_factor) / h_max))
                            .ceil() as u16;
                        component.size.height = (lines
                            * (f32::from(component.vertical_sampling_factor) / v_max))
                            .ceil() as u16;

                        component.block_size.width =
                            frame.mcu_size.width * u16::from(component.horizontal_sampling_factor);
                        component.block_size.height =
                            frame.mcu_size.height * u16::from(component.vertical_sampling_factor);
                    }
                }

                frame
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

struct DQT([u16; 64]);

impl Default for DQT {
    fn default() -> Self {
        DQT([0; 64])
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
            self.0[usize::from(UNZIGZAG[i])] = from.into();
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
    any()
        .and_then(|b| -> Result<_, StreamErrorFor<I>> {
            Ok((
                DQTPrecision::new(b >> 4).ok_or_else(|| {
                    StreamErrorFor::<I>::message_static_message("Unexpected DQT precision")
                })?,
                b & 0x0F,
            ))
        })
        .then_partial(|&mut (precision, identifier)| match precision {
            DQTPrecision::Bit8 => count_min_max(64, 64, any()).left(),
            DQTPrecision::Bit16 => count_min_max(64, 64, be_u16()).right(),
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

fn app_adobe<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

#[derive(Default)]
struct Decoder {
    frame: Option<Frame>,
    scan: Option<Scan>,
    ac_huffman_tables: [Option<huffman::Table>; 16],
    dc_huffman_tables: [Option<huffman::Table>; 16],
}

impl Decoder {
    fn decode(&self) -> Result<(), &'static str> {
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

        for mcu_y in 0..usize::from(frame.mcu_size.height) {
            for mcu_x in 0..usize::from(frame.mcu_size.width) {
                for (i, component) in frame.components.iter().enumerate() {
                    let blocks_per_mcu = usize::from(
                        component.horizontal_sampling_factor * component.vertical_sampling_factor,
                    );
                    for j in 0..blocks_per_mcu {
                        let (block_x, block_y);
                        {
                            let blocks_per_row = usize::from(component.block_size.width);
                            let block_num = (mcu_y * usize::from(frame.mcu_size.width) + mcu_x)
                                * blocks_per_mcu
                                + j;

                            block_x = (block_num % blocks_per_row) as u16;
                            block_y = (block_num / blocks_per_row) as u16;

                            if block_x * 8 >= component.size.width
                                || block_y * 8 >= component.size.height
                            {
                                continue;
                            }
                        }

                        let block_offset = (block_y as usize * component.block_size.width as usize
                            + block_x as usize)
                            * 64;
                        let mcu_row_offset = mcu_y as usize
                            * component.block_size.width as usize
                            * component.vertical_sampling_factor as usize
                            * 64;
                        let coefficients = &mut mcu_row_coefficients[i]
                            [block_offset - mcu_row_offset..block_offset - mcu_row_offset + 64];

                        eprintln!(
                            "{:?}",
                            self.ac_huffman_tables
                                .iter()
                                .map(|t| t.is_some())
                                .collect::<Vec<_>>()
                        );

                        eprintln!("{:?}", scan.headers[i]);
                        let ac_table = self.ac_huffman_tables
                            [usize::from(scan.headers[i].ac_table_selector)]
                        .as_ref()
                        .unwrap_or_else(|| panic!("Missing AC table {}", i)); // TODO un-unwrap
                        self.decode_block(coefficients, &ac_table)
                    }
                }
            }
        }
        Ok(())
    }

    fn decode_block(&self, coefficients: &mut [i16], ac_table: &huffman::Table) {}

    fn do_segment<'a, I>(&mut self, segment: Segment<'a>) -> Result<(), I::Error>
    where
        I: FullRangeStream<Item = u8, Range = &'a [u8]> + From<&'a [u8]> + 'a,
        I::Error: ParseError<I::Item, I::Range, I::Position>,
    {
        log::trace!("Segment {:?}", segment);
        match segment.marker {
            Marker::SOI | Marker::RST(_) | Marker::EOI => Ok(()),
            Marker::SOF(0) => {
                self.frame = Some(sof0().parse(I::from(segment.data))?.0);
                Ok(())
            }
            Marker::SOF(2) => {
                self.frame = Some(sof2().parse(I::from(segment.data))?.0);
                Ok(())
            }
            Marker::DHT => {
                skip_many1(huffman_table().map(|dht| {
                    let table = huffman::Table::new(dht.code_lengths, dht.values).unwrap(); // FIXME
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
            Marker::DQT => dqt().parse(I::from(segment.data)).map(|_| ()),
            Marker::DRI => dri().parse(I::from(segment.data)).map(|_| ()),
            Marker::SOS => {
                let input = I::from(segment.data);
                let frame = self.frame.as_ref().ok_or_else(|| {
                    I::Error::from_error(
                        input.position(),
                        StreamErrorFor::<I>::message_static_message("Found SOS before SOF"),
                    )
                })?;
                self.scan = Some(sos(frame).parse(input)?.0);
                self.decode().map_err(|msg| {
                    let input = I::from(segment.data);
                    I::Error::from_error(
                        input.position(),
                        StreamErrorFor::<I>::message_static_message(msg),
                    )
                })?;
                Ok(())
            }
            Marker::APP_ADOBE => app_adobe().parse(I::from(segment.data)).map(|_| ()),
            Marker::APP(_) => Ok(()),
            _ => panic!("Unhandled segment {:?}", segment.marker),
        }
    }
}

pub fn decode(input: &[u8], output: &mut [u8]) -> Result<(), easy::Errors<String, String, usize>> {
    let mut decoder = Decoder::default();

    let mut parser = many1::<Vec<_>, _>(
        segment().flat_map(|segment| decoder.do_segment::<easy::Stream<&[u8]>>(segment)),
    )
    .skip(eof());
    parser.easy_parse(input).map_err(|err| {
        err.map_position(|pos| pos.translate_position(input))
            .map_token(|token| format!("0x{:X}", token))
            .map_range(|range| format!("{:?}", range))
    })?;

    let frame = decoder.frame.unwrap(); // FIXME
    let component = &frame.components[0];

    let line_stride = usize::from(frame.component_width(0));
    let height = usize::from(frame.lines);
    let width = usize::from(frame.samples_per_line);

    for y in 0..height {
        for x in 0..width {
            // output[y * width + height] = data[y * line_stride + x];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let _ = env_logger::try_init();
        assert_eq!(decode(include_bytes!("../img0.jpg"), &mut [0; 128]), Ok(()));
    }

    #[test]
    fn green() {
        let _ = env_logger::try_init();
        assert_eq!(
            decode(include_bytes!("../tests/images/green.jpg"), &mut [0; 128]),
            Ok(())
        );
    }
}
