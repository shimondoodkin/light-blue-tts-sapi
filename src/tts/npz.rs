//! Minimal NPZ / NPY loader for f32 arrays.
//!
//! NPZ files are ZIP archives containing .npy files.
//! We support only the subset needed: little-endian f32/f64 arrays, C-contiguous.

use std::collections::HashMap;

/// A loaded array: shape + flat f32 data.
#[derive(Debug, Clone)]
pub struct NpArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpArray {
    /// Total number of elements.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Load all arrays from an `.npz` file.
pub fn load_npz(
    path: &std::path::Path,
) -> Result<HashMap<String, NpArray>, Box<dyn std::error::Error + Send + Sync>> {
    let bytes = std::fs::read(path)?;
    let entries = zip_entries(&bytes)?;
    let mut arrays = HashMap::new();
    for (name, data) in entries {
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        let arr = parse_npy(&data)?;
        arrays.insert(key, arr);
    }
    Ok(arrays)
}

// -----------------------------------------------------------------------
// Minimal ZIP reader
// -----------------------------------------------------------------------

fn zip_entries(
    data: &[u8],
) -> Result<Vec<(String, Vec<u8>)>, Box<dyn std::error::Error + Send + Sync>> {
    let len = data.len();
    // Find end-of-central-directory (signature 0x06054b50)
    let eocd_pos = (0..=len.saturating_sub(22))
        .rev()
        .find(|&i| {
            data[i] == 0x50
                && data[i + 1] == 0x4b
                && data[i + 2] == 0x05
                && data[i + 3] == 0x06
        })
        .ok_or("Could not find EOCD record in ZIP")?;

    let cd_offset = read_u32_le(data, eocd_pos + 16) as usize;
    let num_entries = read_u16_le(data, eocd_pos + 10) as usize;

    let mut entries = Vec::new();
    let mut pos = cd_offset;

    for _ in 0..num_entries {
        if pos + 46 > len {
            return Err("Truncated central directory".into());
        }
        if &data[pos..pos + 4] != b"\x50\x4b\x01\x02" {
            return Err("Invalid central directory header".into());
        }
        let compression = read_u16_le(data, pos + 10);
        let compressed_size = read_u32_le(data, pos + 20) as usize;
        let uncompressed_size = read_u32_le(data, pos + 24) as usize;
        let name_len = read_u16_le(data, pos + 28) as usize;
        let extra_len = read_u16_le(data, pos + 30) as usize;
        let comment_len = read_u16_le(data, pos + 32) as usize;
        let local_header_offset = read_u32_le(data, pos + 42) as usize;

        let name = String::from_utf8_lossy(&data[pos + 46..pos + 46 + name_len]).to_string();
        pos = pos + 46 + name_len + extra_len + comment_len;

        // Read from local file header
        if local_header_offset + 30 > len {
            return Err("Truncated local header".into());
        }
        let local_name_len = read_u16_le(data, local_header_offset + 26) as usize;
        let local_extra_len = read_u16_le(data, local_header_offset + 28) as usize;
        let data_start = local_header_offset + 30 + local_name_len + local_extra_len;

        let file_data = match compression {
            0 => {
                // Stored
                data[data_start..data_start + uncompressed_size].to_vec()
            }
            8 => {
                // Deflated
                let compressed = &data[data_start..data_start + compressed_size];
                inflate_raw(compressed, uncompressed_size)?
            }
            _ => {
                return Err(format!("Unsupported ZIP compression method: {}", compression).into());
            }
        };

        entries.push((name, file_data));
    }

    Ok(entries)
}

// -----------------------------------------------------------------------
// Deflate decoder
// -----------------------------------------------------------------------

fn inflate_raw(
    compressed: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut output = Vec::with_capacity(expected_size);
    let mut reader = BitReader::new(compressed);

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => {
                reader.align_to_byte();
                let blen = reader.read_u16_le()?;
                let _nlen = reader.read_u16_le()?;
                for _ in 0..blen {
                    output.push(reader.read_byte()?);
                }
            }
            1 => inflate_block_fixed(&mut reader, &mut output)?,
            2 => inflate_block_dynamic(&mut reader, &mut output)?,
            _ => return Err("Invalid deflate block type".into()),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output)
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bit_count: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    fn ensure_bits(&mut self, n: u8) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        while self.bit_count < n {
            if self.pos >= self.data.len() {
                return Err("Unexpected end of deflate data".into());
            }
            self.bit_buf |= (self.data[self.pos] as u32) << self.bit_count;
            self.pos += 1;
            self.bit_count += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u8) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_bits(n)?;
        let val = self.bit_buf & ((1 << n) - 1);
        self.bit_buf >>= n;
        self.bit_count -= n;
        Ok(val)
    }

    fn align_to_byte(&mut self) {
        self.bit_buf = 0;
        self.bit_count = 0;
    }

    fn read_byte(&mut self) -> Result<u8, Box<dyn std::error::Error + Send + Sync>> {
        if self.pos >= self.data.len() {
            return Err("Unexpected end of data".into());
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_le(&mut self) -> Result<u16, Box<dyn std::error::Error + Send + Sync>> {
        let lo = self.read_byte()? as u16;
        let hi = self.read_byte()? as u16;
        Ok(lo | (hi << 8))
    }

    fn read_huffman(
        &mut self,
        table: &HuffmanTable,
    ) -> Result<u16, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_bits(15)?;
        let mut code: u32 = 0;
        for bits in 1..=15u8 {
            code = (code << 1) | ((self.bit_buf >> (bits - 1)) & 1);
            let reversed = reverse_bits(code, bits);
            if let Some(&sym) = table.codes.get(&(bits, reversed as u16)) {
                self.bit_buf >>= bits;
                self.bit_count -= bits;
                return Ok(sym);
            }
        }
        Err("Invalid Huffman code".into())
    }
}

fn reverse_bits(val: u32, bits: u8) -> u32 {
    let mut result = 0u32;
    let mut v = val;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

struct HuffmanTable {
    codes: HashMap<(u8, u16), u16>,
}

fn build_huffman_table(lengths: &[u8]) -> HuffmanTable {
    let max_bits = *lengths.iter().max().unwrap_or(&0) as usize;
    let mut bl_count = vec![0u32; max_bits + 1];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u32; max_bits + 2];
    let mut code = 0u32;
    for bits in 1..=max_bits {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    let mut codes = HashMap::new();
    for (sym, &l) in lengths.iter().enumerate() {
        if l > 0 {
            let c = next_code[l as usize];
            next_code[l as usize] += 1;
            codes.insert((l, c as u16), sym as u16);
        }
    }
    HuffmanTable { codes }
}

fn make_fixed_lit_lengths() -> Vec<u8> {
    let mut lengths = vec![0u8; 288];
    for i in 0..=143 {
        lengths[i] = 8;
    }
    for i in 144..=255 {
        lengths[i] = 9;
    }
    for i in 256..=279 {
        lengths[i] = 7;
    }
    for i in 280..=287 {
        lengths[i] = 8;
    }
    lengths
}

fn inflate_block_fixed(
    reader: &mut BitReader,
    output: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let lit_lengths = make_fixed_lit_lengths();
    let dist_lengths = vec![5u8; 32];
    let lit_table = build_huffman_table(&lit_lengths);
    let dist_table = build_huffman_table(&dist_lengths);
    inflate_codes(reader, output, &lit_table, &dist_table)
}

fn inflate_block_dynamic(
    reader: &mut BitReader,
    output: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;

    static CODE_LENGTH_ORDER: [usize; 19] =
        [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let mut cl_lengths = vec![0u8; 19];
    for i in 0..hclen {
        cl_lengths[CODE_LENGTH_ORDER[i]] = reader.read_bits(3)? as u8;
    }
    let cl_table = build_huffman_table(&cl_lengths);

    let mut lengths = Vec::with_capacity(hlit + hdist);
    while lengths.len() < hlit + hdist {
        let sym = reader.read_huffman(&cl_table)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let repeat = reader.read_bits(2)? as usize + 3;
                let last = *lengths.last().ok_or("Invalid repeat")?;
                for _ in 0..repeat {
                    lengths.push(last);
                }
            }
            17 => {
                let repeat = reader.read_bits(3)? as usize + 3;
                for _ in 0..repeat {
                    lengths.push(0);
                }
            }
            18 => {
                let repeat = reader.read_bits(7)? as usize + 11;
                for _ in 0..repeat {
                    lengths.push(0);
                }
            }
            _ => return Err("Invalid code length symbol".into()),
        }
    }

    let lit_table = build_huffman_table(&lengths[..hlit]);
    let dist_table = build_huffman_table(&lengths[hlit..hlit + hdist]);
    inflate_codes(reader, output, &lit_table, &dist_table)
}

static LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
    131, 163, 195, 227, 258,
];
static LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
static DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
static DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13,
];

fn inflate_codes(
    reader: &mut BitReader,
    output: &mut Vec<u8>,
    lit_table: &HuffmanTable,
    dist_table: &HuffmanTable,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        let sym = reader.read_huffman(lit_table)?;
        if sym < 256 {
            output.push(sym as u8);
        } else if sym == 256 {
            break;
        } else {
            let len_idx = (sym - 257) as usize;
            if len_idx >= LENGTH_BASE.len() {
                return Err("Invalid length code".into());
            }
            let length =
                LENGTH_BASE[len_idx] as usize + reader.read_bits(LENGTH_EXTRA[len_idx])? as usize;
            let dist_sym = reader.read_huffman(dist_table)? as usize;
            if dist_sym >= DIST_BASE.len() {
                return Err("Invalid distance code".into());
            }
            let distance =
                DIST_BASE[dist_sym] as usize + reader.read_bits(DIST_EXTRA[dist_sym])? as usize;

            let start = output
                .len()
                .checked_sub(distance)
                .ok_or("Invalid back-reference")?;
            for i in 0..length {
                let b = output[start + (i % distance)];
                output.push(b);
            }
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------
// NPY parser
// -----------------------------------------------------------------------

fn parse_npy(data: &[u8]) -> Result<NpArray, Box<dyn std::error::Error + Send + Sync>> {
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic".into());
    }

    let major = data[6];

    let (header_len, header_offset) = if major == 1 {
        (read_u16_le(data, 8) as usize, 10)
    } else {
        (read_u32_le(data, 8) as usize, 12)
    };

    let header_bytes = &data[header_offset..header_offset + header_len];
    let header = String::from_utf8_lossy(header_bytes);

    let is_f64 = header.contains("<f8") || header.contains("float64");
    let is_f32 = header.contains("<f4") || header.contains("float32");
    if !is_f32 && !is_f64 {
        return Err(format!("Unsupported NPY dtype in header: {}", header).into());
    }

    let shape = parse_shape(&header)?;
    let data_start = header_offset + header_len;
    let remaining = &data[data_start..];
    let num_elements: usize = shape.iter().product();

    let float_data: Vec<f32> = if is_f64 {
        let mut floats = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let off = i * 8;
            if off + 8 > remaining.len() {
                return Err("NPY data truncated".into());
            }
            let val = f64::from_le_bytes(remaining[off..off + 8].try_into().unwrap());
            floats.push(val as f32);
        }
        floats
    } else {
        let mut floats = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let off = i * 4;
            if off + 4 > remaining.len() {
                return Err("NPY data truncated".into());
            }
            let val = f32::from_le_bytes(remaining[off..off + 4].try_into().unwrap());
            floats.push(val);
        }
        floats
    };

    Ok(NpArray {
        shape,
        data: float_data,
    })
}

fn parse_shape(header: &str) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync>> {
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("No shape field in NPY header")?;
    let after = &header[shape_start..];
    let paren_start = after.find('(').ok_or("No '(' in shape")? + 1;
    let paren_end = after.find(')').ok_or("No ')' in shape")?;
    let shape_str = &after[paren_start..paren_end];

    let mut dims = Vec::new();
    for part in shape_str.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            dims.push(trimmed.parse::<usize>()?);
        }
    }

    if dims.is_empty() {
        dims.push(1);
    }

    Ok(dims)
}

fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}
