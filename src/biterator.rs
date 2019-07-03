use combine::{
    easy,
    error::StreamError,
    stream::{Positioned, ResetStream, StreamErrorFor, StreamOnce},
    ParseError, Stream,
};

#[derive(Clone, Debug)]
pub(crate) struct Biterator<I> {
    pub(crate) input: I,
    bits: u64,
    count: u8,
}

impl<I> Biterator<I> {
    pub fn new(input: I) -> Self {
        Biterator {
            input,
            bits: 0,
            count: 0,
        }
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
    I: Stream<Item = u8>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
    I::Position: Default,
{
    pub fn count(&self) -> u8 {
        self.count
    }

    pub fn receive_extend(&mut self, count: u8) -> Option<i16> {
        let value = self.next_bits(count)?;
        Some(extend(value, count))
    }

    pub fn reset(&mut self) {
        self.bits = 0;
        self.count = 0;
    }

    pub fn next_bits(&mut self, count: u8) -> Option<u16> {
        if self.count < count {
            self.fill_bits()?;
        }
        if self.count < count {
            return None;
        }
        let bits = self.peek_bits(count);
        self.consume_bits(count);
        Some(bits)
    }

    pub fn consume_bits(&mut self, count: u8) {
        debug_assert!(self.count >= count);
        self.bits <<= count;
        self.count -= count;
    }

    pub fn peek_bits(&self, count: u8) -> u16 {
        debug_assert!(self.count >= count);

        ((self.bits >> (64 - count)) & ((1 << count) - 1)) as u16
    }

    pub fn peek_bits_u8(&self, count: u8) -> u8 {
        debug_assert!(self.count >= count);
        assert!(count <= 8);

        ((self.bits >> (64 - count)) & ((1 << count) - 1)) as u8
    }

    pub fn fill_bits(&mut self) -> Option<()> {
        while self.count <= 56 {
            let checkpoint = self.checkpoint();
            let b = match self.input.uncons().ok()? {
                0xFF => {
                    if self.input.uncons().ok() == Some(0x00) {
                        0xFF
                    } else {
                        ResetStream::reset(self, checkpoint).ok()?;
                        while self.count <= 56 {
                            self.count += 8;
                        }
                        return Some(()); // Not a stuffed 0xFF so we found a marker.
                    }
                }
                b => b,
            };
            self.bits |= u64::from(b) << 56 - self.count;
            self.count += 8;
        }
        Some(())
    }
}

pub fn extend(v: u16, t: u8) -> i16 {
    let vt = 1 << (u16::from(t) - 1);
    if v < vt {
        v as i16 + (-1 << i16::from(t)) + 1
    } else {
        v as i16
    }
}

impl<I> StreamOnce for Biterator<I>
where
    I: Stream<Item = u8>,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
    I::Position: Default,
{
    type Item = bool;
    type Range = u16;
    type Position = I::Position;
    type Error = easy::Errors<bool, u16, I::Position>;

    #[inline]
    fn uncons(&mut self) -> Result<bool, StreamErrorFor<Self>> {
        self.next_bits(1)
            .map(|i| i != 0)
            .ok_or_else(|| StreamErrorFor::<Self>::end_of_input())
    }
}

impl<I> ResetStream for Biterator<I>
where
    I: Stream<Item = u8> + ResetStream,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
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
    I: Stream<Item = u8> + Positioned,
    I::Error: ParseError<I::Item, I::Range, I::Position>,
    I::Position: Default,
{
    fn position(&self) -> Self::Position {
        self.input.position()
    }
}
