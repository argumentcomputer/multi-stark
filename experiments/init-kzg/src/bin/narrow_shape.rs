use multi_stark::{plonkish::{CircuitBuilder,gadgets::{ByteGadgets,blake3},foreign::GoldilocksCircuit},types::Val,ark_adapter::{Scalar,KzgConfig,Srs},system::System,traits::Algebra};
use p3_matrix::dense::RowMajorMatrix;
use std::sync::Arc;
fn main(){
 let mut b=CircuitBuilder::<Val>::new();let bytes=ByteGadgets::new(&mut b);let input=(0..64).map(|i|bytes.input(&mut b,&format!("byte{i}"))).collect::<Vec<_>>();let digest=blake3(&mut b,&bytes,&input);for v in digest{b.expose_public(v.value());}
 let foreign=GoldilocksCircuit::new(&b.finish());let lowered=foreign.circuit.lower_to_multi_stark(Scalar::ONE).unwrap().merge_table_traces(1<<22).unwrap();
 let config=KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2,b"shape-only")),8);
 let mut fields=1;let mut points=3;
 for (i,mut input) in lowered.kzg_circuit_inputs(1usize<<32,8).unwrap().into_iter().enumerate(){
  let p=input.preprocessed.as_ref().unwrap().width;input.preprocessed=Some(RowMajorMatrix::new(vec![Scalar::ZERO;p*2],p));
  let (s,_)=System::new(config.clone(),[input]);let c=&s.circuits[0];let q=c.quotient_degree();
  println!("circuit{i}: advice={} fixed={} stage2={} quotient={} group={}",c.main_width,p,c.stage_2_width,q,c.lookup_group_size);
  points+=(c.main_width+c.stage_2_width+q)*if i==0{1}else{2};fields+=c.main_width+p+2*c.stage_2_width+q;
 }
 println!("Hypothetical compact proof: {} bytes; {points} G1 points; {fields} scalars; quotient domain unsupported at full Init height",5+points*48+fields*32);
}
