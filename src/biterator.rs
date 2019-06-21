use combine::{
    error::UnexpectedParse,
    stream::{PointerOffset, Positioned, ResetStream, StreamErrorFor, StreamOnce},
};

#[derive(Clone, Debug)]
pub(crate) struct Biterator<'a> {
    pub(crate) input: &'a [u8],
    bits: u64,
    count: u8,
}

impl<'a> Biterator<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Biterator {
            input,
            bits: 0,
            count: 0,
        }
    }

    pub fn into_inner(self) -> &'a [u8] {
        self.input
    }

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

    pub fn fill_bits(&mut self) -> Option<()> {
        while self.count <= 56 {
            if self.input.is_empty() {
                return None;
            }
            let b = match self.input[0] {
                0xFF if self.input.get(1) == Some(&0x00) => {
                    self.input = &self.input[2..];
                    0xFF
                }
                0xFF => {
                    while self.count <= 56 {
                        self.count += 8;
                    }
                    return Some(()); // Not a stuffed 0xFF so we found a marker.
                }
                b => {
                    self.input = &self.input[1..];
                    b
                }
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

impl<'a> StreamOnce for Biterator<'a> {
    type Item = bool;
    type Range = u16;
    type Position = PointerOffset<[u8]>;
    type Error = UnexpectedParse;

    #[inline]
    fn uncons(&mut self) -> Result<bool, StreamErrorFor<Self>> {
        self.next_bits(1)
            .map(|i| i != 0)
            .ok_or(UnexpectedParse::Eoi)
    }
}

impl<'a> ResetStream for Biterator<'a> {
    type Checkpoint = Self;

    fn checkpoint(&self) -> Self {
        self.clone()
    }
    fn reset(&mut self, checkpoint: Self) -> Result<(), Self::Error> {
        *self = checkpoint;
        Ok(())
    }
}

impl<'a> Positioned for Biterator<'a> {
    fn position(&self) -> Self::Position {
        self.input.position()
    }
}
