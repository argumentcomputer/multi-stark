"""Rebuild the saved Init FRI proof's key from its historical circuit source.

Run from the repository root; writes only under target/init-fri-wrap.
"""
import pathlib
import subprocess
import tarfile
import io
import re

REV = "296316c852ee71d383305c7f33e53524dc980e26"
root = pathlib.Path("target/init-fri-wrap/historical")
root.mkdir(parents=True, exist_ok=True)
archive = subprocess.check_output(["git", "archive", REV])
with tarfile.open(fileobj=io.BytesIO(archive)) as files:
    files.extractall(root, filter="data")

# Transport derives only; preserve historical circuit/protocol behavior.
for name, types in {
    "graph.rs": ["NodeId", "Node", "ConstraintGraph"],
    "expr.rs": ["Source", "RowOffset", "ColRef"],
    "lookup.rs": ["Lookup"],
}.items():
    path = root / "src" / name
    source = path.read_text()
    for typename in types:
        source, count = re.subn(
            r"(#\[derive\()([^\n]+)(\)\]\npub (?:struct|enum) " + typename + r"\b)",
            r"\1serde::Serialize, serde::Deserialize, \2\3", source,
        )
        assert count == 1, typename
    path.write_text(source)

# Same version-1 verifier-only transport as the current key codec.
codec = pathlib.Path("src/plonkish/verifier/key_codec.rs").read_text()
data = codec[codec.index("#[derive(Serialize, Deserialize)]"):codec.index("impl VerifierKey")]
encode = codec[codec.index("        let s = self.system();"):codec.index("    /// Decode a trusted key")]
encode = encode[:encode.rindex("    }")]
encode = encode.replace("        let s = self.system();\n", "")
encode = encode.replace(".map_err(|e| VerifierError::Encoding(e.to_string()))", ".map_err(|e| e.to_string())")
helper = '''use serde::{Deserialize, Serialize};
use multi_stark::{graph::ConstraintGraph, types::{Val, Commitment, GoldilocksBlake3Config}, system::System};
'''
helper += data + "pub fn encode(s: &System<GoldilocksBlake3Config>) -> Result<Vec<u8>, String> {\n"
helper += encode.replace("crate::config::", "multi_stark::config::") + "}\n"
(root / "examples/support/export_key.rs").write_text(helper)

path = root / "examples/ix_root.rs"
source = path.read_text()
source = source.replace("    system::{System, SystemWitness},", "    system::System,")
source = '#[path = "support/export_key.rs"]\nmod export_key;\n' + source.replace("//! Check/prove", "// Check/prove")
start = source.index("    let start = Instant::now();\n    let mut witness")
end = source.index("    let config = GoldilocksBlake3Config::new(cp, fp);")
source = source[:start] + '''    assert_eq!(circuit.stats().values, 132014830, "historical layout changed");
    let public: Vec<_> = claims.iter().flatten().copied().collect();
    drop(expanded);
    drop(proof);
    drop(system);
    drop(inputs);
    let compiled = circuit.lower_to_multi_stark_with_max_height(Val::from_u8(107), max_trace_height)?;
    let outer_claims = compiled.claims(&public)?;
    let mut wrong = public.clone();
    wrong[0] += Val::ONE;
    let wrong_claims = compiled.claims(&wrong)?;
    let mut definitions = compiled.circuit_inputs();
    for d in &mut definitions { d.lookup_group_size = 3; }
    drop(compiled);
''' + source[end:]
start = source.index("    let refs: Vec<_>")
source = source[:start] + '''    drop(key);
    let bytes = fs::read(out.join("outer-proof.bin"))?;
    let proof = multi_stark::prover::Proof::<GoldilocksBlake3Config>::from_bytes(&bytes)?;
    let refs: Vec<_> = outer_claims.iter().map(Vec::as_slice).collect();
    outer.verify_multiple_claims(&refs, &proof).map_err(|e| format!("saved proof: {e:?}"))?;
    assert!(outer.verify_multiple_claims(&wrong_claims.iter().map(Vec::as_slice).collect::<Vec<_>>(), &proof).is_err());
    fs::write(out.join("outer-vk.bin"), export_key::encode(&outer)?)?;
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&(outer_claims.len() as u64).to_le_bytes());
    for claim in &outer_claims {
        encoded.extend_from_slice(&(claim.len() as u64).to_le_bytes());
        for value in claim { encoded.extend_from_slice(&value.as_canonical_u64().to_le_bytes()); }
    }
    fs::write(out.join("outer-claims.bin"), encoded)?;
    println!("Saved {}-byte proof verified; altered Init claim rejected; exported key", bytes.len());
    Ok(())
}
'''
source = source.replace("PrimeField64, TwoAdicField", "PrimeField64")
path.write_text(source)
print(root)
