use combine::{
    easy,
    parser::{
        byte::{byte, num::be_u16, take_until_byte},
        item::{any, eof, satisfy_map, value},
        range::{take, take_while1},
        repeat::many1,
    },
    stream::FullRangeStream,
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
    take_until_byte(0xFF) // mozjpeg skips any non marker bytes (non 0xFF)
        .skip(take_while1(|b| b == 0xFF)) // Extraenous 0xFF bytes are allowed
        .with(
            satisfy_map(|b| {
                Some(match b {
                    0xD8 => Marker::SOI,
                    0xC4 => Marker::DHT,
                    0xDB => Marker::DQT,
                    0xE0...0xEF => Marker::APP(b - 0xE0),
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
    marker().then_partial(|&mut marker| match marker {
        Marker::SOI => value(Segment { marker, data: &[] }).left(),
        Marker::DHT | Marker::DQT | Marker::APP(_) => be_u16()
            .then(|quantization_table_len| take(quantization_table_len.into()))
            .map(move |data| Segment { marker, data })
            .right(),
        _ => panic!("Unhandled marker {:?}", marker),
    })
}

fn dht<'a, I>() -> impl Parser<Output = (), Input = I>
where
    I: FullRangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    any().map(|b| (b & 0x0F, b >> 4)).with(value(()))
}

fn dqt<'a, I>() -> impl Parser<Output = (), Input = I>
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
    I: FullRangeStream<Item = u8, Range = &'a [u8]> + From<&'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    match segment.marker {
        Marker::SOI => Ok(()),
        Marker::DHT => dht().parse(I::from(segment.data)).map(|_| ()),
        Marker::DQT => dqt().parse(I::from(segment.data)).map(|_| ()),
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
