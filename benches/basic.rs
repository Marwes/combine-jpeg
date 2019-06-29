use std::{fs, path::Path};

use criterion::{criterion_group, criterion_main, Bencher, Benchmark, Criterion};

use combine_jpeg::Decoder;

fn bench_decode(b: &mut Bencher, name: &str) {
    let in_path = Path::new("tests/images").join(name).with_extension("jpg");
    bench_decode_path(b, &in_path, name)
}
fn bench_decode_path(b: &mut Bencher, in_path: &Path, name: &str) {
    let _ = env_logger::try_init();
    let mut decoder = Decoder::default();
    let input =
        fs::read(in_path).unwrap_or_else(|err| panic!("Can't read image {}: {}", name, err));

    b.iter(|| decoder.decode(&input).unwrap());
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

fn decode_group(c: &mut Criterion) {
    c.bench(
        "it_works",
        Benchmark::new("it_works", it_works).sample_size(10),
    );
    c.bench_function("green", green);
    c.bench_function("simple", simple);
}

criterion_group!(decode, decode_group,);
criterion_main!(decode);
