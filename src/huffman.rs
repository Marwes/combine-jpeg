use bitvec::prelude::*;

#[derive(Default)]
pub(crate) struct Table {
    // TODO Use array?
    values: Vec<u8>,
    // TODO i32 seems redundant here
    max_code: [i32; 18],
    val_offset: [i32; 18],
}

impl Table {
    pub(crate) fn new(bits: &[u8; 16], values: &[u8]) -> Result<Table, &'static str> {
        log::trace!("Table::new({:?}, {:?})", bits, values);
        debug_assert!(values.len() <= 256);

        let (huffsize, num_symbols) = huffsize(bits)?;
        log::trace!("huffsize: {:?}", &huffsize[..num_symbols]);

        let mut huffcode = [0u8; 256]; // Size?
        let mut k = 0;
        let mut code = 0;
        let mut si = huffsize[0];

        loop {
            while huffsize[k] == si {
                huffcode[k] = code;
                code += 1;
                k += 1;
            }
            if huffsize[k] == 0 {
                break;
            }
            if u64::from(code) >= 1 << u64::from(si) {
                log::trace!("bad huffman code length: {} {}", code, si);
                return Err("bad huffman code length"); // ?
            }
            code <<= 1;
            si += 1;
        }

        let mut table = Table::default();
        table.values = values.to_owned();

        let mut p = 0i32;
        for l in 0..16 {
            if bits[l] != 0 {
                table.val_offset[l] = p - i32::from(huffcode[(p as usize)]);
                p += i32::from(bits[l]);
                table.max_code[l] = i32::from(huffcode[(p as usize) - 1]);
            } else {
                table.max_code[l] = -1; // No code of this length
            }
        }
        table.max_code[16] = 0;
        table.val_offset[16] = 0xFFFFF;

        Ok(table)
    }
}

fn huffsize(bits: &[u8; 16]) -> Result<([u8; 256], usize), &'static str> {
    let mut huffsize = [0u8; 256];
    let mut p = 0usize;
    for (l, &value) in bits.iter().enumerate() {
        let value = usize::from(value);
        if p + value > 256 {
            return Err("bad huffsize");
        }
        for x in &mut huffsize[p..(p + value)] {
            *x = l as u8 + 1;
        }
        p += value;
    }
    Ok((huffsize, p))
}

struct Decoder;

impl Decoder {
    fn decode(&self, table: &Table, mut input: impl Iterator<Item = bool>) -> Option<u8> {
        let mut code = 0i32;
        let mut offset = 0;
        for (i, &max_code) in table.max_code.iter().enumerate() {
            if code <= max_code {
                offset = i;
                break;
            }
            code <<= 1;
            code |= i32::from(input.next()?);
        }
        Some(table.values[(code + table.val_offset[offset]) as usize])
    }
}

fn extend(v: u16, t: u8) -> i16 {
    let vt = 1 << (u16::from(t) - 1);
    if v < vt {
        v as i16 + (-1 << i16::from(t)) + 1
    } else {
        v as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // TODO
    fn decode() {
        let _ = env_logger::try_init();

        // let (book, tree) = huffman_compress::CodeBuilder::from_iter(vec![("a", 2), ("b", 1)]);
        let mut costs = [0; 16];
        costs[0] = 2;
        costs[1] = 1;
        let values = [1, 2, 3];

        let table = Table::new(&costs, &values).unwrap();
        assert_eq!(
            Decoder.decode(&table, [0b1000_0000u8].as_bitslice::<BigEndian>().iter()),
            Some(1)
        );
    }
}
