use combine::{
    easy,
    parser::{
        byte::{byte, num::be_u16},
        item::{satisfy_map, value},
        range::take,
        repeat::many1,
    },
    ParseError, Parser, RangeStream,
};

#[derive(Debug, Copy, Clone)]
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

fn marker<'a, I>() -> impl Parser<Output = Marker, Input = I>
where
    I: RangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    byte(0xFF).with(satisfy_map(|b| {
        Some(match b {
            0xD8 => Marker::SOI,
            0xDB => Marker::DQT,
            _ => return None,
        })
    }))
}

#[derive(Copy, Clone)]
struct Segment<'a> {
    marker: Marker,
    data: &'a [u8],
}

fn segment<'a, I>() -> impl Parser<Output = Segment<'a>, Input = I>
where
    I: RangeStream<Item = u8, Range = &'a [u8]>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
{
    marker().then_partial(|&mut marker| match marker {
        Marker::SOI => value(Segment { marker, data: &[] }).left(),
        Marker::DQT => be_u16()
            .then(|i| take(i.into()))
            .map(move |data| Segment { marker, data })
            .right(),
        _ => panic!("Unhandled marker {:?}", marker),
    })
}

pub fn decode(input: &[u8], output: &mut [u8]) -> Result<(), easy::Errors<String, String, usize>> {
    let mut parser = many1::<Vec<_>, _>(segment());
    parser.easy_parse(input).map(|_| ()).map_err(|err| {
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
