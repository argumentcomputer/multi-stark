#[path="../storage.rs"] mod storage;
use multi_stark::{ark_adapter::Scalar,plonkish::{CircuitBuilder,gadgets::{ByteGadgets,blake3}},traits::Field};
use std::path::PathBuf;
fn main()->storage::Result<()> {
 let dir=PathBuf::from(std::env::args().nth(1).unwrap());std::fs::create_dir_all(&dir)?;
 let mut b=CircuitBuilder::<Scalar>::new();b.enable_compact_blake3();
 let x=b.input("x");let square=b.mul(x,x);b.expose_public(square);
 let bg=ByteGadgets::new(&mut b);let byte=bg.constant(&mut b,5);let hash=blake3(&mut b,&bg,&[byte]);b.expose_public(hash[0].value());
 let compiled=b.finish().lower_to_multi_stark_sharded(Scalar::from_u8(93),1<<16)?;
 let mut witness=compiled.witness();witness.set(x,Scalar::from_u8(3))?;let assignment=witness.generate()?;
 let shards=compiled.trace_shards(&assignment)?;
 let mut manifest=storage::Manifest {widths:vec![],heights:vec![],claims:compiled.claims(assignment.public_values())?};
 for i in 0..compiled.num_circuits() {
  let mut d=compiled.kzg_circuit_input(i,1<<16,8)?.unwrap();let fixed=d.preprocessed.take().unwrap();
  manifest.widths.push(d.main_width);manifest.heights.push(fixed.values.len()/fixed.width);
  storage::save(&dir.join(format!("{i}.meta")),&d)?;
  storage::write_matrix(&dir.join(format!("{i}.fixed.zst")),&fixed)?;
  storage::write_matrix(&dir.join(format!("{i}.witness.zst")),&shards.trace(i)?)?;
 }
 storage::save(&dir.join("manifest.bin"),&manifest)?;println!("Fixture staged");Ok(())
}
