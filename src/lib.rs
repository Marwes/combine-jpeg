use combine::{
    parser::{byte::byte, item::satisfy_map},
    ParseError, Parser, RangeStream,
};

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
            _ => return None,
        })
    }))
}

pub fn decode(input: &[u8], output: &mut [u8]) -> Result<(), ()> {
    let mut parser = marker();
    parser.parse(input).map(|_| ()).map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(decode(include_bytes!("../img0.jpg"), &mut [0; 128]).is_ok());
    }
}
