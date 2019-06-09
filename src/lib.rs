type JPEG = ();

pub fn decode(input: &[u8]) -> Result<JPEG, ()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(decode(include_bytes!("../img0.jpg")).is_ok());
    }
}
