# Backend interfaces

`ProofConfig` selects the field, transcript, domain and polynomial commitment
scheme through `traits`. `p3_adapter` supplies the Goldilocks FRI implementation.
`StarkGenericConfig` is a compatibility alias for `ProofConfig`.

External backends implement these traits without changing the proving system.
Their transcript must bind the parameters and any optional opening omissions.
The domain and quotient limits must match the backend's actual capabilities.

- `CircuitInputs::compile_graph` inspects constraints before setup.
- `System::new_without_preprocessed` commits fixed matrices without
  retaining a second copy. `prove_committed` accepts previously committed traces.
- `ProofConfig` has optional fused lookup and quotient commitment hooks; they
  must preserve the portable path's constraints, ordering and commitments.
- `compute_lookup_values_range` evaluates chunks with full-trace row semantics.
- `observe_claims` and `sample_lookup_challenges` share transcript conventions
  with external verifier implementations; neither verifies a proof.

See [the Plonkish API](plonkish.md) for circuit construction and translation.
