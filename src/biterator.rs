use combine::{
    easy,
    error::StreamError,
    stream::{Positioned, ResetStream, StreamErrorFor, StreamOnce},
    ParseError, Stream,
};

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Biterator<I> {
    pub(crate) input: I,
    pub(crate) bits: u64,
    pub(crate) count: u8,
}

impl<I> Biterator<I> {
    pub fn new(input: I) -> Self {
        Self::with_state(input, 0, 0)
    }

    pub fn with_state(input: I, bits: u64, count: u8) -> Self {
        Biterator { input, bits, count }
    }

    pub fn as_inner(&self) -> &I {
        &self.input
    }

    pub fn as_inner_mut(&mut self) -> &mut I {
        &mut self.input
    }
}

impl<I> Biterator<I>
where
    I: Stream<Token = u8>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Position: Default,
{
    pub fn count(&self) -> u8 {
        self.count
    }

    pub fn receive_extend(&mut self, count: u8) -> Result<i16, ()> {
        let value = self.next_bits(count)?;
        Ok(extend(value, count))
    }

    pub fn reset(&mut self) {
        self.bits = 0;
        self.count = 0;
    }

    pub fn next_bits(&mut self, count: u8) -> Result<u16, ()> {
        if self.count < count {
            self.fill_bits()?;
        }
        let bits = self.peek_bits(count);
        self.consume_bits(count);
        Ok(bits)
    }

    pub fn consume_bits(&mut self, count: u8) {
        debug_assert!(self.count >= count);
        self.count -= count;
    }

    pub fn peek_bits(&self, count: u8) -> u16 {
        debug_assert!(self.count >= count);

        ((self.bits >> (self.count - count)) & ((1 << count) - 1)) as u16
    }

    pub fn peek_bits_u8(&self, count: u8) -> u8 {
        debug_assert!(self.count >= count);
        debug_assert!(count <= 8);

        ((self.bits >> (self.count - count)) & ((1 << count) - 1)) as u8
    }

    pub fn fill_bits(&mut self) -> Result<(), ()> {
        let checkpoint = self.checkpoint();
        while self.count <= 56 {
            let b = self.input.uncons().map_err(|_| ())?;
            self.bits = self.bits << 8 | u64::from(b);
            self.count += 8;

            if b == 0xFF {
                if self.input.uncons().map_err(|_| ())? != 0x00 {
                    ResetStream::reset(&mut self.input, checkpoint.0).map_err(|_| ())?;
                    for _ in (checkpoint.2..self.count - 8).step_by(8) {
                        self.input.uncons().map_err(|_| ())?;
                    }
                    self.bits &= !0xFF; // Replace the byte we read with zero bits

                    // And fill the rest up with zeroes as well
                    while self.count <= 56 {
                        self.bits <<= 8;
                        self.count += 8;
                    }
                    return Ok(()); // Not a stuffed 0xFF so we found a marker.
                }
            }
        }
        Ok(())
    }
}

pub fn extend(v: u16, t: u8) -> i16 {
    // Branchless extend copied from https://github.com/mozilla/mozjpeg
    let vt = i32::from(1 << (u16::from(t) - 1));
    let v = i32::from(v);
    (v + (((v - vt) >> 31) as u32 & (((-1i32 as u32) << t) + 1)) as i32) as i16
}

impl<I> StreamOnce for Biterator<I>
where
    I: Stream<Token = u8>,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Position: Default,
{
    type Token = bool;
    type Range = u16;
    type Position = I::Position;
    type Error = easy::Errors<bool, u16, I::Position>;

    #[inline]
    fn uncons(&mut self) -> Result<bool, StreamErrorFor<Self>> {
        self.next_bits(1)
            .map(|i| i != 0)
            .map_err(|()| StreamErrorFor::<Self>::end_of_input())
    }

    fn is_partial(&self) -> bool {
        self.input.is_partial()
    }
}

impl<I> ResetStream for Biterator<I>
where
    I: Stream<Token = u8> + ResetStream,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Position: Default,
{
    type Checkpoint = (I::Checkpoint, u64, u8);

    fn checkpoint(&self) -> Self::Checkpoint {
        (self.input.checkpoint(), self.bits, self.count)
    }
    fn reset(&mut self, checkpoint: Self::Checkpoint) -> Result<(), Self::Error> {
        self.input.reset(checkpoint.0).map_err(|_| {
            Self::Error::from_error(
                self.position(),
                StreamErrorFor::<Self>::message_static_message("Unable to reset"),
            )
        })?;
        self.bits = checkpoint.1;
        self.count = checkpoint.2;
        Ok(())
    }
}

impl<I> Positioned for Biterator<I>
where
    I: Stream<Token = u8> + Positioned,
    I::Error: ParseError<I::Token, I::Range, I::Position>,
    I::Position: Default,
{
    fn position(&self) -> Self::Position {
        self.input.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut biterator = Biterator::new(&[0b11010111, 0b00110010, 0xFF, 0xAB][..]);

        assert_eq!(
            (0..)
                .scan((), |_, _| biterator.uncons().ok())
                .take(16)
                .collect::<Vec<_>>(),
            [
                true, true, false, true, false, true, true, true, false, false, true, true, false,
                false, true, false
            ]
        );
    }

    #[test]
    fn multibit() {
        let mut biterator = Biterator::new(&[0b11010111, 0b00110010, 0xFF, 0xAB][..]);

        assert_eq!(
            (0..)
                .scan((), |_, _| biterator.next_bits(3).ok())
                .take(5)
                .collect::<Vec<_>>(),
            [0b110, 0b101, 0b110, 0b011, 0b001]
        );
    }

    #[test]
    #[ignore]
    fn multibyte() {
        let mut biterator =
            Biterator::new(&[0b11010111, 0b00110010, 0b11010111, 0b00110010, 0xFF, 0xAB][..]);

        assert_eq!(
            (0..)
                .scan((), |_, _| biterator.next_bits(9).ok())
                .collect::<Vec<_>>(),
            [0b110101110, 0b011001011, 0b0101110010,]
        );
    }
}
