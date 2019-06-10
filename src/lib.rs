use combine::{
    easy,
    error::{Consumed, StreamError},
    parser::{
        byte::{byte, num::be_u16, take_until_byte},
        function::parser,
        item::{any, eof, satisfy_map, value},
        range::{take, take_while1},
        repeat::{count_min_max, many1, sep_by1},
    },
    stream::{FullRangeStream, StreamErrorFor},
    ParseError, Parser,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Marker {
    SOI,
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

#[derive(Copy, Clone)]
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

fn sof0<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

fn sof2<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

struct HuffmanTable {
    values: [u8; 16],
}

fn dht<'a, I>() -> impl Parser<Output = (), Input = I> + 'a
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    (any().map(|b| (b & 0x0F, b >> 4)), take(16))
        .then_partial(
            |&mut ((table_class, destination), code_lengths): &mut (_, &'a [u8])| {
                parser(move |input: &mut I| {
                    for &len in code_lengths {
                        let (input_slice, _) = take(len.into()).parse_lazy(input).into_result()?;
                    }
                    Ok(((), Consumed::Consumed(())))
                })
            },
        )
        .with(value(()))
}

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
        for (to, from) in self.0.iter_mut().zip(iter) {
            *to = from.into();
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
                DQTPrecision::new(b & 0x0F).ok_or_else(|| {
                    StreamErrorFor::<I>::message_static_message("Unexpected DQT precision")
                })?,
                b >> 4,
            ))
        })
        .then_partial(|&mut (precision, identifier)| match precision {
            DQTPrecision::Bit8 => count_min_max(64, 64, be_u16()).left(),
            DQTPrecision::Bit16 => count_min_max(64, 64, any()).right(),
        })
}

fn dri<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

fn sos<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

fn app_adobe<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

fn do_segment<'a, I>(segment: Segment<'a>) -> Result<(), I::Error>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + From<&'a [u8]> + 'a,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    match segment.marker {
        Marker::SOI | Marker::RST(_) | Marker::EOI => Ok(()),
        Marker::SOF(0) => sof0().parse(I::from(segment.data)).map(|_| ()),
        Marker::SOF(2) => sof2().parse(I::from(segment.data)).map(|_| ()),
        Marker::DHT => dht().parse(I::from(segment.data)).map(|_| ()),
        Marker::DQT => dqt().parse(I::from(segment.data)).map(|_| ()),
        Marker::DRI => dri().parse(I::from(segment.data)).map(|_| ()),
        Marker::SOS => sos().parse(I::from(segment.data)).map(|_| ()),
        Marker::APP_ADOBE => app_adobe().parse(I::from(segment.data)).map(|_| ()),
        _ => panic!("Unhandled segment {:?}", segment.marker),
    }
}

pub fn decode(input: &[u8], output: &mut [u8]) -> Result<(), easy::Errors<String, String, usize>> {
    let mut parser = many1::<Vec<_>, _>(
        segment().flat_map(|segment| do_segment::<easy::Stream<&[u8]>>(segment)),
    )
    .skip(eof());
    parser
        .easy_parse(input)
        .map(|(_, _rest)| ())
        .map_err(|err| {
            err.map_position(|pos| pos.translate_position(input))
                .map_token(|token| format!("0x{:X}", token))
                .map_range(|range| format!("{:?}", range))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(decode(include_bytes!("../img0.jpg"), &mut [0; 128]), Ok(()));
    }
}
