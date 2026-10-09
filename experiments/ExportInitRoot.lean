import Ix.Aggr.Backend
import Ix.Aggr.Verification
import Ix.Store

def main (args : List String) : IO UInt32 := do
  let [address, output] := args
    | throw <| IO.userError "usage: export-init-root <root-address> <output-dir>"
  let address ← match Address.fromString address with
    | some value => pure value
    | none => throw <| IO.userError "expected a 64-character root address"
  let backend ← match ← Aggr.buildVerificationBackend Aggr.defaultRecursionParameters with
    | .ok value => pure value
    | .error error => throw <| IO.userError error
  let bytes ← StoreIO.toIO <| Store.read address
  let wrapper ← match Aggr.decodeAggregateWrapperAt address bytes with
    | .ok value => pure value
    | .error error => throw <| IO.userError error
  let proof ← match Aiur.Proof.ofBytesChecked wrapper.proof with
    | .ok value => pure value
    | .error error => throw <| IO.userError error
  let claim := Aggr.aggregateOuterClaim backend.allowed backend.aggrIdx wrapper.claim
  match backend.system.verify claim proof with
  | .error error => throw <| IO.userError s!"root verification: {error}"
  | .ok () => pure ()
  let output := System.FilePath.mk output
  IO.FS.createDirAll output
  IO.FS.writeBinFile (output / "root-vk.bin") backend.system.vkBytes
  IO.FS.writeBinFile (output / "root-proof.bin") wrapper.proof
  IO.FS.writeBinFile (output / "root-claims.bin") (MultiStark.serializeClaims #[claim])
  IO.println s!"Exported verified root {address}: {wrapper.proof.size} proof bytes, {claim.size} claim words"
  return 0
