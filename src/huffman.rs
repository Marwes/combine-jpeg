use std::iter;

use arrayvec::ArrayVec;
use combine::{ParseError, Stream};

use crate::biterator::{extend, Biterator};

#[derive(Copy, Clone, Debug)]
pub enum TableClass {
    DC = 0,
    AC = 1,
}

impl TableClass {
    pub fn new(b: u8) -> Option<Self> {
        Some(match b {
            0 => TableClass::DC,
            1 => TableClass::AC,
            _ => return None,
        })
    }
}

const LUT_BITS: u8 = 8;

pub(crate) struct BaseTable {
    values: [u8; 256],
    max_code: [i32; 16],
    val_offset: [i32; 16],
    lut: [(u8, u8); 1 << LUT_BITS],
}

impl Default for BaseTable {
    fn default() -> Self {
        BaseTable {
            values: [0; 256],
            max_code: Default::default(),
            val_offset: Default::default(),
            lut: [(0, 0); 1 << LUT_BITS],
        }
    }
}

pub(crate) type DcTable = BaseTable;

pub(crate) struct AcTable {
    table: BaseTable,
    ac_lut: [(i16, u8); 1 << LUT_BITS],
}

impl Default for AcTable {
    fn default() -> Self {
        AcTable {
            table: Default::default(),
            ac_lut: [(0, 0); 1 << LUT_BITS],
        }
    }
}

impl BaseTable {
    pub(crate) fn new(bits: &[u8; 16], values: &[u8]) -> Result<Self, &'static str> {
        log::trace!("Table::new({:?}, {:?})", bits, values);
        debug_assert!(values.len() <= 256);

        let huffsize = huffsize(bits)?;
        log::trace!("huffsize: {:?}", huffsize);

        let mut huffcode = [0u16; 256];
        let mut code = 0u32;
        let mut code_size = huffsize[0];

        for (&size, huffcode_elem) in huffsize.iter().zip(&mut huffcode[..huffsize.len()]) {
            while code_size < size {
                code <<= 1;
                code_size += 1;
            }

            if u64::from(code) >= 1 << u64::from(code_size) {
                log::trace!("bad huffman code length: {} {}", code, code_size);
                return Err("bad huffman code length"); // ?
            }

            *huffcode_elem = code as u16;
            code += 1;
        }

        let mut table = BaseTable::default();
        table.values[..values.len()].copy_from_slice(values);

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

        // Build a lookup table for faster decoding.
        for (i, &size) in huffsize
            .iter()
            .enumerate()
            .filter(|&(_, &size)| size <= LUT_BITS)
        {
            let bits_remaining = LUT_BITS - size;
            let start = usize::from(huffcode[i] << bits_remaining);

            let v = (values[i], size);
            for slot in &mut table.lut[start..start + (1 << bits_remaining)] {
                *slot = v;
            }
        }

        Ok(table)
    }

    pub(crate) fn decode<I>(&self, input: &mut Biterator<I>) -> Option<u8>
    where
        I: Stream<Token = u8>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        I::Position: Default,
    {
        if input.count() < 16 {
            input.fill_bits()?;
            if input.count() < LUT_BITS {
                return None;
            }
        }
        let (value, size) = self.lut[usize::from(input.peek_bits(LUT_BITS))];

        if size > 0 {
            input.consume_bits(size);
            Some(value)
        } else {
            if input.count() < 16 {
                return None;
            }
            let bits = input.peek_bits(16);

            for i in LUT_BITS..16 {
                let code = i32::from(bits >> (15 - i));

                if code <= self.max_code[usize::from(i)] {
                    input.consume_bits(i + 1);

                    let index = (code + self.val_offset[usize::from(i)]) as usize;
                    return Some(self.values[index]);
                }
            }
            None
        }
    }
}

impl AcTable {
    pub(crate) fn new(bits: &[u8; 16], values: &[u8]) -> Result<Self, &'static str> {
        let table = BaseTable::new(bits, values)?;

        let mut ac_lut = [(0i16, 0u8); 1 << LUT_BITS];

        for (i, &(value, size)) in table.lut.iter().enumerate() {
            let run_length = value >> 4;
            let magnitude_category = value & 0x0f;

            if magnitude_category > 0 && size + magnitude_category <= LUT_BITS {
                let unextended_ac_value = (((i << size) & ((1 << LUT_BITS) - 1))
                    >> (LUT_BITS - magnitude_category))
                    as u16;
                let ac_value = extend(unextended_ac_value, magnitude_category);

                ac_lut[i] = (ac_value, (run_length << 4) | (size + magnitude_category));
            }
        }

        Ok(AcTable { table, ac_lut })
    }

    pub(crate) fn decode<I>(&self, input: &mut Biterator<I>) -> Option<u8>
    where
        I: Stream<Token = u8>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        I::Position: Default,
    {
        self.table.decode(input)
    }

    pub(crate) fn decode_fast_ac<I>(
        &self,
        input: &mut Biterator<I>,
    ) -> Result<Option<(i16, u8)>, ()>
    where
        I: Stream<Token = u8>,
        I::Error: ParseError<I::Token, I::Range, I::Position>,
        I::Position: Default,
    {
        if input.count() < LUT_BITS {
            input.fill_bits().ok_or_else(|| ())?;
            if input.count() < LUT_BITS {
                return Ok(None);
            }
        }

        let (value, run_size) = self.ac_lut[usize::from(input.peek_bits_u8(LUT_BITS))];

        if run_size != 0 {
            let run = run_size >> 4;
            let size = run_size & 0x0f;

            input.consume_bits(size);
            return Ok(Some((value, run)));
        }

        Ok(None)
    }
}

fn huffsize(bits: &[u8; 16]) -> Result<ArrayVec<[u8; 256]>, &'static str> {
    let mut huffsize = ArrayVec::new();
    for (l, &value) in bits.iter().enumerate() {
        let value = usize::from(value);
        if huffsize.len() + value > 256 {
            return Err("bad huffsize");
        }
        huffsize.extend(iter::repeat(l as u8 + 1).take(value));
    }
    Ok(huffsize)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestTable {
        bits: &'static [u8; 16],
        values: &'static [u8],
        table_class: TableClass,
    }

    // Tables defined in section K of https://www.w3.org/Graphics/JPEG/itu-t81.pdf
    // Copied from https://github.com/kaksmet/jpeg-decoder

    const TABLE_K3: TestTable = TestTable {
        bits: &[
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ],
        values: &[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ],
        table_class: TableClass::DC,
    };
    const TABLE_K4: TestTable = TestTable {
        bits: &[
            0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ],
        values: &[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ],
        table_class: TableClass::DC,
    };
    const TABLE_K5: TestTable = TestTable {
        bits: &[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D,
        ],
        values: &[
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ],
        table_class: TableClass::AC,
    };
    const TABLE_K6: TestTable = TestTable {
        bits: &[
            0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ],
        values: &[
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ],
        table_class: TableClass::AC,
    };

    fn run_test(test_table: &TestTable) {
        match test_table.table_class {
            TableClass::AC => {
                AcTable::new(&test_table.bits, &test_table.values).unwrap();
            }
            TableClass::DC => {
                DcTable::new(&test_table.bits, &test_table.values).unwrap();
            }
        }
    }

    #[test]
    fn table_k3() {
        run_test(&TABLE_K3);
    }

    #[test]
    fn table_k4() {
        run_test(&TABLE_K4);
    }

    #[test]
    fn table_k5() {
        run_test(&TABLE_K5);
    }

    #[test]
    fn table_k6() {
        run_test(&TABLE_K6);
    }
}
