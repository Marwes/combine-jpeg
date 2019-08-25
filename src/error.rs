use std::io;

use {
    combine::easy,
    derive_more::{Display, From},
};

#[derive(Debug, PartialEq, Eq, Display)]
pub enum UnsupportedFeature {
    NonIntegerSubsamplingRatio,
}

#[derive(Debug, PartialEq, Display, From)]
pub enum Error {
    #[display(fmt = "{}", _0)]
    Unsupported(UnsupportedFeature),

    #[display(fmt = "{}", _0)]
    Parse(easy::Errors<String, String, usize>),

    #[display(fmt = "{}", _0)]
    Format(String),

    #[display(fmt = "{}", _0)]
    Message(&'static str),
}

#[derive(Debug, Display)]
pub enum IoError {
    #[display(fmt = "{}", _0)]
    Error(Error),

    #[display(fmt = "{}", _0)]
    IO(io::Error),
}

impl<E> From<E> for IoError
where
    Error: From<E>,
{
    fn from(err: E) -> Self {
        IoError::Error(err.into())
    }
}

impl From<io::Error> for IoError {
    fn from(err: io::Error) -> Self {
        IoError::IO(err)
    }
}
