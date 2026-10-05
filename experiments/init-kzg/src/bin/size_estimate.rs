#[path="../storage.rs"] mod storage;
use std::{path::PathBuf,sync::Arc,process::{Command,Stdio},io::Read};
use multi_stark::{ark_adapter::{KzgConfig,KzgCommitment,KzgProof,Scalar,Srs},system::{System,CircuitInputs},batch::{BatchProof,BatchPreamble,ShardHeader},prover::{Proof,Commitments},traits::Algebra,expr::{Source,RowOffset},graph::Node};
use p3_matrix::dense::RowMajorMatrix;
fn main()->storage::Result<()> {
 let dir=PathBuf::from(std::env::args().nth(1).unwrap());
 let mut m:storage::Manifest=storage::load(&dir.join("manifest.bin"))?;
 let collapse=std::env::args().nth(2).as_deref()==Some("unpartitioned");
 if collapse {m.heights[0]=1<<30;}
 let count=m.widths.len();let max_height=*m.heights.iter().max().unwrap();
 let srs=Arc::new(Srs::unsafe_dev_setup(2,b"size-only-placeholder"));let point=srs.g1[0];let config=KzgConfig::new(srs,8);
 let mut headers=vec![];let mut proofs=vec![];
 let mut totals=[0usize;4];let mut group_bytes=[0usize;3];
 for i in 0..count {
  if collapse && (1..61).contains(&i) {continue}
  let mut child=Command::new("zstd").args(["-q","-d","-c"]).arg(dir.join(format!("{i}.fixed.zst"))).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
  let mut buf=[0u8;16];child.stdout.as_mut().unwrap().read_exact(&mut buf)?;let mut p=usize::try_from(u64::from_le_bytes(buf[..8].try_into()?))?;let _=child.kill();child.wait()?;
  let mut input:CircuitInputs<Scalar>=storage::load(&dir.join(format!("{i}.meta")))?;
  if collapse && i==0 {p-=1;input.lookups.pop();}
  input.preprocessed=Some(RowMajorMatrix::new(vec![Scalar::ZERO;p*2],p));
  let (sys,_)=System::new(config.clone(),[input]);let c=&sys.circuits[0];
  let next=|s|usize::from(c.graph.nodes.iter().any(|n|matches!(n,Node::Var(col) if col.source==s && col.offset==RowOffset::Next)));
  let w=c.main_width;let s=c.stage_2_width;let q=c.quotient_degree();
  for (sum,v) in totals.iter_mut().zip([w,p,s,q]){*sum+=v;}
  let commit=|width|KzgCommitment(vec![vec![point;width]],vec![vec![point;if m.heights[i]<max_height{width}else{0}]]);
  let mut active=vec![false;count];active[i]=true;let logs=vec![m.heights[i].ilog2() as u8];
  let header=ShardHeader{active:active.clone(),stage_1_trace:commit(w),log_degrees:logs.clone(),claims:if i==0{m.claims.clone()}else{vec![]}};
  let mut prep=vec![vec![];count];prep[i]=vec![vec![Scalar::ZERO;p];1+next(Source::Preprocessed)];
  let proof=Proof::<KzgConfig>{active,log_degrees:logs,commitments:Commitments{stage_1_trace:commit(w),stage_2_trace:commit(s),quotient_chunks:commit(q)},intermediate_accumulators:vec![Scalar::ZERO],opening_proof:KzgProof(vec![point;2]),quotient_opened_values:vec![vec![vec![Scalar::ZERO;q]]],preprocessed_opened_values:Some(prep),stage_1_opened_values:vec![vec![vec![Scalar::ZERO;w];1+next(Source::Main)]],stage_2_opened_values:vec![vec![vec![Scalar::ZERO;s];2]]};
  let hb=bincode::serde::encode_to_vec(&header,bincode::config::standard())?.len();let pb=proof.to_bytes()?.len();
  group_bytes[if i<61{0}else if i<65{1}else{2}]+=hb+pb;
  println!("shard {i}: main={w} fixed={p} lookup={s} quotient={q} header_bytes={hb} proof_bytes={pb}");
  headers.push(header);proofs.push(proof);
 }
 let mut single=proofs[0].clone();
 single.active=vec![];single.log_degrees=vec![];single.intermediate_accumulators=vec![];
 single.commitments=Commitments{stage_1_trace:KzgCommitment(vec![],vec![]),stage_2_trace:KzgCommitment(vec![],vec![]),quotient_chunks:KzgCommitment(vec![],vec![])};
 single.stage_1_opened_values.clear();single.stage_2_opened_values.clear();single.quotient_opened_values.clear();single.preprocessed_opened_values=Some(vec![]);
 for proof in &proofs {
  single.active.push(true);single.log_degrees.extend(&proof.log_degrees);single.intermediate_accumulators.push(Scalar::ZERO);
  for (dst,src) in [&mut single.commitments.stage_1_trace,&mut single.commitments.stage_2_trace,&mut single.commitments.quotient_chunks].into_iter().zip([&proof.commitments.stage_1_trace,&proof.commitments.stage_2_trace,&proof.commitments.quotient_chunks]) {dst.0.extend(src.0.clone());dst.1.extend(src.1.clone());}
  single.stage_1_opened_values.extend(proof.stage_1_opened_values.clone());single.stage_2_opened_values.extend(proof.stage_2_opened_values.clone());single.quotient_opened_values.extend(proof.quotient_opened_values.clone());
  single.preprocessed_opened_values.as_mut().unwrap().extend(proof.preprocessed_opened_values.as_ref().unwrap().iter().filter(|v|!v.is_empty()).cloned());
 }
 let mut heights=single.log_degrees.clone();heights.sort();heights.dedup();single.opening_proof=KzgProof(vec![point;heights.len()+1]);
 let points=[&single.commitments.stage_1_trace,&single.commitments.stage_2_trace,&single.commitments.quotient_chunks].into_iter().flat_map(|c|c.0.iter().chain(&c.1).flatten()).count()+single.opening_proof.0.len();
 let fields=[&single.stage_1_opened_values,&single.stage_2_opened_values,&single.quotient_opened_values,single.preprocessed_opened_values.as_ref().unwrap()].into_iter().flatten().flatten().flatten().count()+single.intermediate_accumulators.len()-1;
 println!("SINGLE ORDINARY PROOF SIZE-ONLY TEMPLATE: {} bytes ordinary, {} compact; {} circuits; {} group points; {} scalars; public statement supplied separately",single.to_bytes()?.len(),5+48*points+32*fields,single.active.len(),points,fields);
 let batch=BatchProof{preamble:BatchPreamble{headers,messages:vec![]},proofs};
 println!("SIZE-ONLY TEMPLATE, NOT A VALID PROOF: {} bytes; column totals {:?}; group bytes {:?}",batch.to_bytes()?.len(),totals,group_bytes);
 Ok(())
}
