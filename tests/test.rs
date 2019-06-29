use combine_jpeg::*;

use std::{fs, iter, path::Path};

fn test_decode(name: &str) -> Vec<u8> {
    let in_path = Path::new("tests/images").join(name).with_extension("jpg");
    test_decode_path(&in_path, name)
}

fn test_decode_path(in_path: &Path, name: &str) -> Vec<u8> {
    let _ = env_logger::try_init();
    let mut decoder = Decoder::default();
    let input =
        fs::read(in_path).unwrap_or_else(|err| panic!("Can't read image {}: {}", name, err));
    let out = decoder.decode(&input).unwrap();

    let frame = decoder.frame.as_ref().expect("Frame");

    let image = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
        u32::from(frame.samples_per_line),
        u32::from(frame.lines),
        &out[..],
    )
    .expect("Image");
    image
        .save(Path::new("target").join(name).with_extension("png"))
        .unwrap();
    out
}

#[test]
fn it_works() {
    let _ = env_logger::try_init();
    let out = test_decode_path(Path::new("img0.jpg"), "img0");
    assert_eq!(
        &out[..100],
        &[
            0, 20, 45, 0, 20, 45, 1, 21, 46, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45,
            0, 20, 45, 0, 20, 45, 1, 21, 46, 1, 21, 46, 1, 21, 46, 1, 21, 46, 0, 20, 45, 0, 20, 45,
            0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45,
            0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45, 0, 20, 45,
            0, 20, 45, 0
        ][..]
    );
}

#[test]
fn green() {
    let out = test_decode("green");

    assert_eq!(
        out,
        iter::repeat(&[35u8, 177, 77])
            .take(16 * 8)
            .flat_map(|xs| &xs[..])
            .cloned()
            .collect::<Vec<_>>()
    );
}

#[test]
fn simple() {
    let out = test_decode("simple");

    assert_eq!(
        out,
        vec![
            246, 255, 254, 246, 255, 254, 251, 255, 255, 244, 249, 252, 249, 255, 255, 248, 255,
            255, 246, 255, 254, 250, 255, 252, 253, 254, 249, 253, 252, 247, 251, 252, 246, 255,
            255, 251, 253, 255, 252, 253, 253, 253, 253, 251, 254, 255, 254, 255, 255, 255, 250,
            255, 254, 250, 252, 255, 255, 245, 252, 255, 242, 255, 255, 246, 255, 255, 252, 253,
            255, 255, 254, 252, 255, 255, 251, 252, 253, 248, 255, 251, 255, 255, 253, 255, 252,
            253, 255, 252, 255, 255, 247, 254, 247, 251, 255, 249, 253, 255, 252, 252, 255, 255,
            252, 255, 255, 250, 252, 251, 252, 255, 255, 239, 252, 245, 237, 255, 246, 236, 255,
            247, 236, 255, 245, 246, 255, 249, 251, 255, 250, 254, 255, 250, 250, 251, 246, 254,
            254, 252, 253, 253, 253, 4, 2, 5, 2, 0, 1, 3, 0, 0, 1, 0, 0, 2, 2, 4, 3, 7, 10, 0, 8,
            9, 245, 253, 255, 255, 254, 255, 255, 251, 248, 253, 252, 248, 252, 252, 250, 254, 255,
            255, 249, 253, 254, 249, 253, 254, 249, 255, 253, 250, 255, 250, 248, 253, 247, 253,
            255, 252, 255, 254, 250, 255, 254, 249, 255, 251, 245, 254, 254, 244, 241, 255, 242,
            196, 235, 206, 119, 175, 140, 172, 227, 195, 221, 255, 237, 231, 255, 239, 248, 255,
            248, 2, 0, 1, 3, 1, 4, 1, 0, 2, 5, 3, 8, 3, 1, 4, 4, 0, 3, 4, 0, 0, 9, 5, 2, 2, 1, 0,
            0, 2, 0, 0, 1, 0, 7, 0, 0, 9, 0, 0, 255, 253, 253, 254, 255, 255, 239, 254, 249, 238,
            255, 247, 248, 255, 253, 249, 255, 249, 246, 253, 246, 251, 255, 252, 254, 255, 255,
            255, 255, 253, 255, 251, 248, 255, 252, 247, 253, 254, 240, 232, 255, 231, 103, 153,
            115, 74, 150, 101, 77, 160, 108, 160, 229, 185, 0, 32, 1, 0, 21, 2, 0, 7, 0, 0, 3, 0,
            1, 3, 2, 3, 4, 6, 0, 0, 4, 1, 0, 4, 4, 0, 1, 4, 0, 0, 6, 3, 0, 4, 6, 0, 9, 11, 0, 4, 0,
            0, 7, 0, 0, 1, 0, 0, 0, 13, 4, 189, 224, 204, 131, 173, 149, 187, 216, 196, 239, 255,
            243, 251, 255, 253, 250, 252, 251, 255, 255, 255, 248, 253, 255, 250, 254, 255, 255,
            255, 253, 249, 255, 245, 230, 255, 233, 91, 148, 105, 63, 153, 92, 59, 164, 97, 59,
            160, 94, 54, 142, 84, 9, 70, 26, 0, 26, 0, 0, 13, 0, 0, 10, 0, 0, 4, 0, 0, 3, 2, 0, 1,
            4, 4, 4, 6, 2, 1, 0, 1, 2, 0, 0, 3, 0, 0, 4, 0, 0, 3, 0, 0, 3, 0, 0, 10, 0, 0, 21, 0,
            70, 135, 93, 80, 156, 109, 87, 145, 107, 218, 255, 231, 247, 255, 252, 255, 254, 255,
            254, 254, 255, 249, 255, 255, 251, 255, 255, 252, 253, 255, 244, 251, 244, 235, 255,
            238, 165, 212, 178, 79, 156, 104, 63, 159, 96, 56, 161, 92, 58, 159, 91, 71, 157, 96,
            66, 136, 84, 22, 74, 35, 0, 25, 0, 0, 24, 0, 0, 17, 0, 0, 11, 0, 0, 8, 0, 0, 4, 0, 0,
            11, 3, 0, 5, 0, 0, 8, 0, 0, 12, 2, 0, 15, 0, 0, 21, 0, 3, 62, 18, 63, 153, 91, 58, 154,
            90, 85, 159, 110, 215, 255, 232, 240, 255, 250, 255, 254, 255, 254, 254, 255, 251, 254,
            255, 249, 253, 255, 245, 249, 252, 251, 255, 255, 248, 255, 250, 234, 255, 240, 218,
            255, 230, 41, 99, 59, 77, 157, 104, 57, 153, 90, 61, 164, 95, 57, 160, 91, 65, 160, 96,
            77, 162, 103, 83, 162, 107, 158, 224, 178, 212, 255, 223, 226, 255, 236, 227, 255, 238,
            233, 255, 243, 244, 255, 253, 0, 7, 0, 0, 16, 0, 5, 59, 23, 62, 139, 83, 63, 160, 91,
            57, 170, 92, 60, 163, 90, 75, 149, 98, 215, 255, 231, 237, 254, 246, 253, 254, 255,
            255, 255, 255, 251, 255, 255, 251, 255, 255, 251, 255, 255, 248, 252, 253, 251, 253,
            252, 249, 255, 251, 242, 255, 247, 0, 19, 0, 0, 31, 0, 39, 111, 65, 67, 156, 98, 59,
            158, 94, 63, 164, 96, 57, 156, 89, 65, 162, 93, 70, 159, 95, 78, 153, 97, 99, 165, 117,
            168, 228, 190, 211, 255, 234, 220, 255, 238, 220, 255, 237, 179, 237, 197, 66, 145, 88,
            61, 163, 90, 52, 165, 85, 53, 164, 87, 63, 155, 90, 142, 203, 159, 226, 255, 235, 246,
            255, 250, 254, 255, 255, 255, 255, 255, 247, 255, 255, 243, 252, 249, 253, 255, 252,
            255, 255, 253, 253, 253, 251, 253, 255, 250, 249, 255, 251, 0, 5, 0, 0, 12, 0, 0, 19,
            0, 0, 25, 0, 48, 103, 64, 80, 148, 101, 72, 153, 95, 67, 157, 93, 64, 161, 92, 61, 161,
            89, 62, 163, 93, 64, 163, 99, 82, 175, 118, 122, 211, 155, 52, 141, 83, 67, 162, 94,
            59, 161, 87, 49, 160, 81, 60, 166, 92, 70, 155, 96, 36, 97, 55, 225, 255, 234, 242,
            255, 245, 249, 254, 247, 255, 255, 253, 255, 255, 255, 245, 254, 253, 251, 255, 255,
            255, 255, 253, 250, 249, 245, 255, 254, 250, 255, 254, 250, 254, 255, 250, 254, 255,
            251, 0, 2, 0, 0, 3, 0, 0, 12, 0, 0, 20, 0, 0, 25, 0, 0, 40, 0, 38, 104, 56, 71, 150,
            93, 65, 160, 92, 56, 157, 87, 61, 160, 93, 58, 159, 93, 56, 163, 93, 57, 164, 92, 59,
            161, 87, 65, 163, 90, 72, 164, 97, 18, 97, 40, 0, 33, 0, 226, 255, 236, 242, 255, 247,
            253, 255, 250, 252, 251, 247, 253, 249, 246, 255, 255, 253, 254, 249, 255, 255, 250,
            254, 254, 252, 253, 255, 255, 253, 255, 254, 251, 255, 254, 251, 255, 254, 250, 255,
            252, 247, 4, 3, 0, 7, 10, 3, 0, 10, 1, 0, 7, 0, 0, 17, 3, 0, 17, 0, 0, 15, 0, 0, 22, 0,
            25, 80, 40, 85, 148, 101, 85, 149, 99, 71, 146, 90, 66, 161, 97, 62, 159, 92, 73, 155,
            93, 81, 148, 95, 41, 89, 49, 0, 22, 0, 0, 11, 0, 246, 255, 250, 253, 255, 254, 255,
            253, 254, 255, 254, 253, 255, 253, 252, 255, 254, 255, 255, 248, 255, 255, 249, 254,
            255, 254, 255, 253, 253, 253, 255, 255, 255, 255, 252, 253, 255, 250, 251, 255, 252,
            253, 253, 249, 246, 2, 3, 0, 0, 2, 0, 2, 7, 0, 1, 3, 0, 0, 1, 0, 0, 4, 0, 3, 16, 6, 0,
            18, 0, 0, 23, 0, 0, 26, 0, 0, 35, 0, 14, 83, 36, 75, 147, 97, 40, 95, 53, 0, 25, 0, 0,
            13, 0, 0, 3, 0, 255, 255, 255, 255, 251, 255, 250, 245, 249, 255, 253, 255, 253, 249,
            250, 255, 252, 253, 255, 254, 255, 255, 247, 248, 255, 251, 252, 254, 255, 255, 247,
            253, 253, 243, 246, 251, 255, 254, 255, 254, 249, 255, 255, 250, 255, 255, 254, 255,
            253, 253, 251, 254, 253, 249, 12, 9, 4, 4, 0, 0, 7, 6, 2, 0, 6, 2, 0, 5, 0, 0, 7, 1, 0,
            8, 3, 0, 10, 0, 0, 10, 0, 0, 16, 0, 0, 20, 0, 0, 18, 2, 0, 10, 0, 0, 3, 0, 250, 254,
            253, 252, 253, 255, 255, 253, 255, 255, 252, 255, 255, 252, 255, 251, 249, 252, 255,
            255, 255, 255, 255, 255, 255, 247, 247, 255, 254, 251, 243, 247, 246, 249, 255, 255,
            251, 255, 255, 250, 251, 255, 252, 249, 255, 254, 252, 255, 251, 252, 254, 255, 255,
            255, 254, 250, 249, 252, 246, 246, 255, 250, 251, 4, 0, 0, 3, 8, 4, 0, 6, 2, 0, 6, 8,
            0, 2, 5, 3, 3, 3, 5, 5, 3, 0, 3, 0, 1, 7, 3, 250, 254, 253, 251, 253, 252, 253, 254,
            255, 254, 255, 255, 251, 255, 255, 249, 249, 249, 255, 248, 251, 254, 245, 248, 255,
            254, 255, 252, 253, 255, 255, 255, 255, 255, 255, 253, 250, 252, 249, 250, 255, 251,
            249, 254, 250, 252, 254, 251, 254, 254, 252, 255, 254, 255, 254, 254, 254, 249, 253,
            254, 249, 253, 254, 252, 253, 255, 255, 255, 255, 255, 253, 255, 255, 252, 255, 254,
            254, 255, 247, 249, 248, 252, 252, 252, 255, 255, 255, 251, 253, 250, 253, 255, 250,
            255, 255, 253, 250, 250, 248, 252, 254, 253, 254, 255, 255, 249, 250, 252, 253, 254,
            255, 247, 253, 253, 252, 255, 255, 255, 255, 253, 255, 251, 250, 252, 252, 250, 253,
            255, 252, 255, 255, 255, 249, 255, 253, 251, 255, 255, 251, 255, 252, 253, 255, 250,
            252, 253, 247, 252, 251, 247, 254, 254, 252, 254, 255, 255, 251, 255, 255, 249, 254,
            255, 251, 254, 255, 253, 254, 255, 253, 251, 255, 255, 251, 255, 255, 254, 255, 255,
            254, 255, 255, 254, 255, 253, 252, 250, 253, 255, 250, 253, 255, 250, 251, 250, 248,
            255, 255, 253, 254, 255, 255, 252, 254, 253, 255, 255, 255, 248, 249, 251, 249, 255,
            255, 248, 254, 252, 248, 248, 246, 255, 254, 250, 255, 255, 251, 253, 255, 252, 255,
            255, 255
        ]
    );
}
