use std::{fs, path::Path};

use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};

use combine_jpeg::Decoder;

fn bench_decode(b: &mut Bencher, name: &str) {
    let in_path = Path::new("tests/images").join(name).with_extension("jpg");
    bench_decode_path(b, &in_path, name)
}
fn bench_decode_path(b: &mut Bencher, in_path: &Path, name: &str) {
    let _ = env_logger::try_init();
    let input =
        fs::read(in_path).unwrap_or_else(|err| panic!("Can't read image {}: {}", name, err));

    b.iter(|| {
        let mut decoder = Decoder::default();
        let out = decoder.decode(black_box(&input)).unwrap();
        out
    });
}

fn it_works(b: &mut Bencher) {
    bench_decode_path(b, Path::new("img0.jpg"), "img0");
}

fn green(b: &mut Bencher) {
    bench_decode(b, "green");
}

fn simple(b: &mut Bencher) {
    bench_decode(b, "simple");
}

fn it_works_mozjpeg(b: &mut Bencher) {
    let name = "img0.jpg";
    let input = fs::read(Path::new(name))
        .unwrap_or_else(|err| panic!("Can't read image {}: {}", name, err));

    b.iter(|| {
        let decompress = mozjpeg::Decompress::new_mem(black_box(&input)).unwrap();
        let mut started = decompress.rgb().unwrap();
        let out = started.read_scanlines::<[u8; 3]>().unwrap();
        assert!(started.finish_decompress());
        out
    })
}

fn it_works_jpeg_decoder(b: &mut Bencher) {
    let name = "img0.jpg";
    let input = fs::read(Path::new(name))
        .unwrap_or_else(|err| panic!("Can't read image {}: {}", name, err));

    b.iter(|| {
        let mut decoder = jpeg_decoder::Decoder::new(black_box(&input[..]));
        let out = decoder.decode().unwrap();
        out
    })
}

fn decode_group(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("it_works");
        group.sample_size(10);

        group.warm_up_time(std::time::Duration::from_millis(1)); // FIXME Remove

        group.bench_function("combine", it_works);
        group.bench_function("mozjpeg", it_works_mozjpeg);
        group.bench_function("jpeg-decoder", it_works_jpeg_decoder);
    }
    c.bench_function("green", green);
    c.bench_function("simple", simple);
}

criterion_group!(decode, decode_group,);
criterion_main!(decode);
