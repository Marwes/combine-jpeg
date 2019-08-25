use std::mem;

use combine::{
    error::ParseResult,
    stream::{
        user_state::StateStream, FullRangeStream, Positioned, RangeStreamOnce, ResetStream,
        StreamErrorFor, StreamOnce,
    },
};

use crate::{biterator::Biterator, Decoder};

#[repr(transparent)]
pub struct DecoderStream<'s, I>(pub BiteratorStream<'s, I>);

pub(crate) fn decoder_stream<'s, 't, I>(
    s: &'t mut BiteratorStream<'s, I>,
) -> &'t mut DecoderStream<'s, I> {
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
    I: Send + StreamOnce,
{
    type Item = I::Item;
    type Range = I::Range;
    type Position = I::Position;
    type Error = I::Error;

    #[inline]
    fn uncons(&mut self) -> Result<I::Item, StreamErrorFor<Self>> {
        self.0.stream.as_inner_mut().uncons()
    }

    fn is_partial(&self) -> bool {
        self.0.stream.as_inner().is_partial()
    }
}

impl<'s, I> ResetStream for DecoderStream<'s, I>
where
    I: Send + ResetStream,
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
    I: Send + RangeStreamOnce,
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
    I: Send + FullRangeStream,
{
    fn range(&self) -> Self::Range {
        self.stream.as_inner().range()
    }
}

impl<'s, I> Positioned for DecoderStream<'s, I>
where
    I: Send + Positioned,
{
    fn position(&self) -> Self::Position {
        self.0.stream.as_inner().position()
    }
}

impl<'s, I> DecoderStream<'s, I> {
    pub fn new(state: &'s mut Decoder, stream: I) -> Self {
        DecoderStream(BiteratorStream {
            state,
            stream: Biterator::new(stream),
        })
    }
}

#[doc(hidden)]
pub type BiteratorStream<'s, I> = StateStream<Biterator<I>, &'s mut Decoder>;
