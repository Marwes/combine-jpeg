#[derive(Default)]
struct Table {
    // TODO i32 seems redundant here
    max_code: [i32; 18],
    val_offset: [i32; 18],
}

fn huffcode(bits: &[u8; 17]) -> Result<Table, ()> {
    let (huffsize, num_symbols) = huffsize(bits)?;

    let mut huffcode = [0u8; 257]; // Size?
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
            return Err(());
        }
        code <<= 1;
        si += 1;
    }

    let mut table = Table::default();

    let mut p = 0i32;
    for l in 1..=16 {
        if bits[l] != 0 {
            table.val_offset[l] = p - i32::from(huffcode[(p as usize)]);
            p += i32::from(bits[l]);
            table.max_code[l] = i32::from(huffcode[(p as usize) - 1]);
        } else {
            table.max_code[l] = -1; // No code of this length
        }
    }
    table.max_code[17] = 0;
    table.val_offset[17] = 0xFFFFF;

    Ok(table)
}

fn huffsize(bits: &[u8; 17]) -> Result<([u8; 257], usize), ()> {
    let mut huffsize = [0u8; 257];
    let mut p = 0usize;
    for (l, &i) in bits.iter().enumerate().skip(1) {
        if p + usize::from(i) > 256 {
            return Err(());
        }
        for x in &mut huffsize[p..(p + usize::from(i))] {
            *x = l as u8;
        }
        p += usize::from(i);
    }
    huffsize[p] = 0;
    Ok((huffsize, p))
}

struct Decoder;

impl Decoder {
    fn decode_ac(&self, input: &[u8]) {
        let mut k = 1;
        let zz = [0; 64];
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
