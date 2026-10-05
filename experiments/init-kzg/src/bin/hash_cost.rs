use multi_stark::{plonkish::{CircuitBuilder,gadgets::{ByteGadgets,blake3},foreign::GoldilocksCircuit},types::Val};
fn main(){
 for len in [32,64,128,1024] {
  for compact in [false,true] {
   let mut b=CircuitBuilder::<Val>::new();if compact{b.enable_compact_blake3();}
   let bytes=ByteGadgets::new(&mut b);
   let input=(0..len).map(|i|{let x=b.input(format!("byte-{i}"));bytes.constrain_byte(&mut b,x)}).collect::<Vec<_>>();
   let h=blake3(&mut b,&bytes,&input);for byte in h {b.expose_public(byte.value());}
   let c=b.finish();println!("len={len} compact={compact} source={:?} translated={:?}",c.stats(),GoldilocksCircuit::estimate(&c));
  }
 }
}
