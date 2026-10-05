#[path="../storage.rs"] mod storage;
use std::{fs,path::PathBuf,sync::Arc,time::{Duration,Instant}};
use multi_stark::ark_adapter::{KzgConfig,Srs,sharded::ShardedKzg};
fn main()->storage::Result<()> {
 let dir=PathBuf::from(std::env::args().nth(1).ok_or("directory")?);
 let checkpoint=dir.join("kzg");fs::create_dir_all(&checkpoint)?;
 let start=Instant::now();
 let config=KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(1<<24,b"init-root-full-kzg-sharded-v1")),8);
 println!("SRS ready: {:?}",start.elapsed());
 for i in 3..6 {
  while !dir.join(format!("{i}.fixed.zst")).exists() {std::thread::sleep(Duration::from_secs(5));}
  let start=Instant::now();
  let mut local=ShardedKzg::new(config.clone(),[storage::definition(&dir,i)?]);
  let data=(local.system.circuits.remove(0),local.system.preprocessed_commit.take());
  storage::save(&checkpoint.join(format!("setup-{i}.bin")),&data)?;
  println!("Setup {i}: {:?}",start.elapsed());
 }
 Ok(())
}
