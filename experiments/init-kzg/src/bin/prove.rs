#[path="../storage.rs"] mod storage;
use std::{fs,path::PathBuf,sync::Arc,time::Instant};
use multi_stark::{ark_adapter::{KzgConfig,KzgCommitment,Scalar,Srs,sharded::ShardedKzg},system::{System,Circuit},batch::{BatchPreamble,BatchProof,ShardHeader},prover::Proof,traits::Algebra};
use p3_matrix::dense::RowMajorMatrix;
fn main()->storage::Result<()> {
    std::thread::spawn(|| loop {std::thread::sleep(std::time::Duration::from_secs(30));if let Ok(s)=fs::read_to_string("/proc/self/status") {eprintln!("{}",s.lines().filter(|l|l.starts_with("VmRSS:")||l.starts_with("VmSize:")).collect::<Vec<_>>().join(" "));}});
    let dir=PathBuf::from(std::env::args().nth(1).ok_or("supply staged directory")?);
    let mode=std::env::var("KZG_RUN_MODE").unwrap_or_else(|_| "all".into());
    let worker:usize=std::env::var("KZG_RUN_WORKER").unwrap_or_else(|_| "0".into()).parse()?;
    let workers:usize=std::env::var("KZG_RUN_WORKERS").unwrap_or_else(|_| "1".into()).parse()?;
    if mode == "all" {
        let status=std::process::Command::new("python3").arg("target/init-kzg-run/workers.py").arg(&dir).status()?;
        if !status.success() {return Err("worker pipeline failed".into())}
        return Ok(());
    }
    let manifest:storage::Manifest=storage::load(&dir.join("manifest.bin"))?;
    let count=manifest.widths.len();
    let height=*manifest.heights.iter().max().ok_or("empty manifest")?;
    let checkpoint=dir.join("kzg");fs::create_dir_all(&checkpoint)?;
    fs::write(checkpoint.join("SECURITY.txt"),"Development SRS: trapdoor known. This run measures correctness and cost, not production security. Inner proof unchanged.\n")?;
    let start=Instant::now();
    let config=KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(height,b"init-root-full-kzg-sharded-v1")),8);
    println!("Development SRS degree {height}: {:?}",start.elapsed());
    let mut circuits=Vec::new();let mut indices=Vec::new();let mut commitment=KzgCommitment(vec![],vec![]);
    for i in 0..count {
        if mode == "setup" && i % workers != worker {continue}
        let start=Instant::now();let file=checkpoint.join(format!("setup-{i}.bin"));
        let (circuit,commit):(Circuit<Scalar>,Option<KzgCommitment>)=if file.exists() {storage::load(&file)?} else {
            if mode != "setup" {return Err("missing setup checkpoint".into())}
            let mut local=ShardedKzg::new(config.clone(),[storage::definition(&dir,i)?]);
            let data=(local.system.circuits.remove(0),local.system.preprocessed_commit.take());
            storage::save(&file,&data)?;data
        };
        if circuit.preprocessed.is_some() || circuit.main_width!=manifest.widths[i] || circuit.preprocessed_height!=manifest.heights[i] {return Err("setup metadata mismatch".into())}
        circuits.push(circuit);
        if let Some(mut c)=commit {indices.push(Some(commitment.0.len()));commitment.0.append(&mut c.0);commitment.1.append(&mut c.1);} else {indices.push(None);}
        println!("Setup {i}/{count}: {:?}",start.elapsed());
    }
    if mode == "setup" {return Ok(())}
    let mut prover=ShardedKzg { system:System {config,circuits,preprocessed_indices:indices,preprocessed_commit:Some(commitment)} };
    let mut claims=vec![vec![];count];claims[0]=manifest.claims.clone();
    let schedule:Vec<_>=(0..count).map(|i|vec![i]).collect();
    let trace=|i:usize|->storage::Result<Vec<RowMajorMatrix<Scalar>>> {
        let mut traces:Vec<_>=manifest.widths.iter().map(|&w|RowMajorMatrix::new(vec![],w)).collect();
        traces[i]=storage::read_matrix(&dir.join(format!("{i}.witness.zst")))?;Ok(traces)
    };
    let mut headers=Vec::new();
    for i in 0..count {
        if mode == "headers" && i % workers != worker {continue}
        let start=Instant::now();let file=checkpoint.join(format!("header-{i}.bin"));
        let header:ShardHeader<KzgConfig>=if file.exists() {storage::load(&file)?} else {
            if mode != "headers" {return Err("missing header checkpoint".into())}
            let h=prover.commit_shard(&claims[i],&schedule[i],trace(i)?)?;storage::save(&file,&h)?;h
        };
        let active:Vec<_>=(0..count).map(|j|j==i).collect();
        if header.active!=active || header.claims!=claims[i] || header.log_degrees!=vec![manifest.heights[i].ilog2() as u8] {return Err("header policy mismatch".into())}
        headers.push(header);println!("Header {i}/{count}: {:?}",start.elapsed());
    }
    if mode == "headers" {return Ok(())}
    let preamble=BatchPreamble {headers,messages:vec![]};
    if mode == "verify" {storage::save(&checkpoint.join("preamble.bin"),&preamble)?;}
    let mut proofs=Vec::new();
    for i in 0..count {
        if mode == "prove" && i % workers != worker {continue}
        let start=Instant::now();let file=checkpoint.join(format!("proof-{i}.bin"));
        let proof:Proof<KzgConfig>=if file.exists() {Proof::from_bytes(&fs::read(&file)?)?} else {
            if mode != "prove" {return Err("missing proof checkpoint".into())}
            let p=prover.prove_shard(&preamble,i,|ci|storage::definition(&dir,ci).expect("staged preprocessing"),trace(i)?)?;
            prover.system.verify_batch_shard(&preamble,i,&p).map_err(|e|format!("shard {i}: {e:?}"))?;
            let temp=file.with_extension("partial");fs::write(&temp,p.to_bytes()?)?;fs::rename(temp,&file)?;p
        };
        prover.system.verify_batch_shard(&preamble,i,&proof).map_err(|e|format!("cached shard {i}: {e:?}"))?;
        proofs.push(proof);println!("PROVED AND VERIFIED {i}/{count}: {:?}",start.elapsed());
    }
    if mode == "prove" {return Ok(())}
    let batch=BatchProof {preamble,proofs};prover.verify(&batch,&claims,&schedule)?;
    let bytes=batch.to_bytes()?;let decoded=BatchProof::from_bytes(&bytes)?;prover.verify(&decoded,&claims,&schedule)?;
    let mut wrong=claims.clone();wrong[0][1][3]+=Scalar::ONE;
    assert!(prover.verify(&decoded,&wrong,&schedule).is_err());
    fs::write(checkpoint.join("proof.bin"),&bytes)?;
    fs::write(checkpoint.join("VERIFIED.txt"),format!("Staged circuit wrapped in {count} KZG shards; {} bytes; expected claims verified; altered expected claim rejected. Development SRS only.\n",bytes.len()))?;
    println!("KZG BATCH VERIFIED: {} bytes, {count} shards; altered claim rejected; development SRS",bytes.len());
    Ok(())
}
