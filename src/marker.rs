use combine::{
    error::ParseError,
    parser::{
        byte::{byte, take_until_byte},
        combinator::{any_send_partial_state, AnySendPartialState},
        range::take_while1,
        repeat::sep_by1,
        token::satisfy_map,
    },
    stream::RangeStream,
    Parser,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Marker {
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
    COM,
    EOI,
}

impl Marker {
    pub const APP_ADOBE: Marker = Marker::APP(14);
}

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
        iter.into_iter().for_each(|_| ())
    }
}

pub fn marker<'a, I>(
) -> impl Parser<I, Output = Marker, PartialState = AnySendPartialState> + crate::Captures<'a>
where
    I: Send + RangeStream<Token = u8, Range = &'a [u8]> + 'a,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
{
    any_send_partial_state(
        sep_by1::<Sink, _, _, _>(
            (
                take_until_byte(0xFF).map(|_| ()), // mozjpeg skips any non marker bytes (non 0xFF)
                take_while1(|b| b == 0xFF).map(|_| ()), // Extraenous 0xFF bytes are allowed
            ),
            byte(0x00).expected("stuffed zero"), // When we encounter a 0x00, we found a stuffed zero (FF/00) sequence so we search again
        )
        .with(
            satisfy_map(|b| {
                Some(match b {
                    0xD8 => Marker::SOI,
                    0xC0..=0xC2 => Marker::SOF(b - 0xC0),
                    0xC4 => Marker::DHT,
                    0xDB => Marker::DQT,
                    0xDD => Marker::DRI,
                    0xDA => Marker::SOS,
                    0xD0..=0xD7 => Marker::RST(b - 0xD0),
                    0xE0..=0xEF => Marker::APP(b - 0xE0),
                    0xD9 => Marker::EOI,
                    0xFE => Marker::COM,
                    _ => return None,
                })
            })
            .expected("marker"),
        ),
    )
}
