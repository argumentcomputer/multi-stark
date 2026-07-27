//! Synthesis of the logUp stage-2 constraints from a circuit's lookups.
//!
//! Replaces the imperative constraint generation in `LookupAir::eval` with a
//! frontend-to-frontend function: given the lookups, it produces the
//! extension-field constraints for the running-accumulator argument, as
//! ordinary [`ExtExpr`] data. `System::new` appends these to the spec's
//! `ext_constraints` before compilation, so they are interned, coordinate-
//! expanded and canonicalized like any other constraint.
//!
//! # Layout conventions (owned here)
//!
//! - **Publics** (each an extension value, stored as `d` base coordinates):
//!   slot 0 = β (lookup challenge), 1 = γ (fingerprint challenge),
//!   2 = current accumulator, 3 = next accumulator. So `num_publics = 4·d`.
//! - **Stage-2 columns** (extension values, flattened to base): slot 0 = the
//!   running accumulator, slots `1..=L` = the message inverse of each lookup,
//!   in lookup order. So `stage2_width = (1 + L)·d` base columns.

use p3_field::Field;

use super::expr::{Expr, ExtExpr, Lookup, RowOffset};

/// Number of extension-valued public inputs the lookup argument uses:
/// β, γ, current accumulator, next accumulator.
pub const LOOKUP_PUBLIC_SIZE: usize = 4;

/// Public-input width (base coordinates) for a circuit whose extension
/// degree is `d`.
pub fn num_publics(d: usize) -> usize {
    LOOKUP_PUBLIC_SIZE * d
}

/// Stage-2 trace width (flattened base columns) for `num_lookups` lookups
/// at extension degree `d`: one accumulator column plus one inverse column
/// per lookup.
pub fn stage2_width(num_lookups: usize, d: usize) -> usize {
    (1 + num_lookups) * d
}

/// Builds the logUp stage-2 constraints for `lookups` at extension degree
/// `d`. Order: the per-lookup message/inverse constraints in lookup order,
/// then the first-row, transition, and last-row accumulator constraints.
pub fn synthesize_lookups<F: Field>(lookups: &[Lookup<Expr<F>>], d: usize) -> Vec<ExtExpr<F>> {
    let d = u32::try_from(d).expect("extension degree exceeds u32");
    let beta = ExtExpr::public(0, d);
    let gamma = ExtExpr::public(1, d);
    let acc = ExtExpr::public(2, d);
    let next_acc = ExtExpr::public(3, d);
    let acc_col = ExtExpr::stage2(0, d, RowOffset::Current);
    let next_acc_col = ExtExpr::stage2(0, d, RowOffset::Next);

    let mut constraints = Vec::with_capacity(lookups.len() + 3);
    // acc_expr = acc_col + Σ_j multiplicity_j · inv_j, built once and reused
    // by the transition and last-row constraints (the interner shares it).
    let mut acc_expr = acc_col.clone();
    for (j, lookup) in lookups.iter().enumerate() {
        let slot = 1 + u32::try_from(j).expect("lookup count exceeds u32");
        let inv = ExtExpr::stage2(slot, d, RowOffset::Current);

        // fingerprint = Σ_i args[i] · γ^i, via Horner over the reversed args.
        let mut coeffs = lookup.args.iter().rev();
        let mut fingerprint = match coeffs.next() {
            Some(arg) => ExtExpr::from(arg.clone()),
            None => ExtExpr::from(Expr::constant(F::ZERO)),
        };
        for arg in coeffs {
            fingerprint = fingerprint * gamma.clone() + arg.clone();
        }

        // message = β + fingerprint; constraint: message · inv − 1 = 0.
        let message = beta.clone() + fingerprint;
        constraints.push(message * inv.clone() - Expr::constant(F::ONE));

        acc_expr = acc_expr + lookup.multiplicity.clone() * inv;
    }

    // First row: acc_col = acc.
    constraints.push(Expr::IsFirstRow * (acc_col - acc));
    // Transition: acc_expr = next row's accumulator column.
    constraints.push(Expr::IsTransition * (acc_expr.clone() - next_acc_col));
    // Last row: acc_expr = next_acc.
    constraints.push(Expr::IsLastRow * (acc_expr - next_acc));

    constraints
}
