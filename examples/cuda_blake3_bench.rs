#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("enable --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::hint::black_box;
    use std::time::Instant;

    use multi_stark::cuda::blake3_hash_rows;
    use p3_blake3::Blake3;
    use p3_maybe_rayon::prelude::*;
    use p3_symmetric::CryptographicHasher;

    println!("backend,message_bytes,messages,iteration,seconds,megabytes");
    for (message_bytes, message_count) in [(64usize, 1 << 20), (4264, 1 << 15), (7400, 1 << 14)] {
        let messages: Vec<u8> = (0..message_bytes * message_count)
            .map(|index| (index as u64).wrapping_mul(0x9e37_79b9).to_le_bytes()[0])
            .collect();
        let expected: Vec<[u8; 32]> = messages
            .par_chunks_exact(message_bytes)
            .map(|message| Blake3.hash_iter(message.iter().copied()))
            .collect();
        let warm = blake3_hash_rows(0, &messages, message_bytes);
        assert_eq!(warm, expected);

        let megabytes =
            f64::from(u32::try_from(messages.len()).expect("benchmark input exceeds u32"))
                / 1_000_000.0;
        for iteration in 0..3 {
            let start = Instant::now();
            let cpu: Vec<[u8; 32]> = messages
                .par_chunks_exact(message_bytes)
                .map(|message| Blake3.hash_iter(message.iter().copied()))
                .collect();
            black_box(&cpu);
            println!(
                "cpu,{message_bytes},{message_count},{iteration},{:.9},{megabytes:.3}",
                start.elapsed().as_secs_f64()
            );

            let start = Instant::now();
            let gpu = blake3_hash_rows(0, &messages, message_bytes);
            let elapsed = start.elapsed().as_secs_f64();
            black_box(&gpu);
            assert_eq!(gpu, expected);
            println!(
                "cuda,{message_bytes},{message_count},{iteration},{:.9},{megabytes:.3}",
                elapsed
            );
        }
    }
}
