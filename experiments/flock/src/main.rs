mod adapter;
mod binary;
mod counts;
#[cfg(feature = "terminal")]
mod fixed_matrices;
#[cfg(feature = "terminal")]
mod fixed_sigma;
#[cfg(feature = "terminal")]
mod fixed_tables;
mod fold;
mod fri;
mod goldilocks;
mod hash_chain;
mod init;
mod packed;
mod parameters;
mod static_wiring;
#[cfg(feature = "terminal")]
mod terminal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let initial_k = if let Some(index) = args.iter().position(|arg| arg == "--initial-k") {
        if !matches!(
            args.first().map(String::as_str),
            Some("init-prove" | "terminal-init")
        ) {
            return Err("--initial-k is only supported for Init proving".into());
        }
        let k: usize = args
            .get(index + 1)
            .ok_or("missing initial-k value")?
            .parse()?;
        if !(4..=6).contains(&k) {
            return Err("expected initial-k 4, 5 or 6".into());
        }
        args.drain(index..=index + 1);
        Some(k)
    } else {
        None
    };
    match args.first().map(String::as_str) {
        #[cfg(feature = "terminal")]
        Some("terminal-gadgets") if args.len() == 2 => terminal::gadgets(args[1].as_ref()),
        #[cfg(feature = "terminal")]
        Some("terminal-fri") if args.len() == 2 => fri::run_inner(args[1].as_ref(), true, true),
        #[cfg(feature = "terminal")]
        Some("terminal-init") if (3..=5).contains(&args.len()) && args.get(4).is_none_or(|s| s == "--coalesced") => init::run_inner(args[1].as_ref(), args[2].as_ref(), Some(match args.get(3).map(String::as_str).unwrap_or("fast") {
            "fast" => flock_prover::pcs::ligerito::LigeritoProfile::Fast,
            "slim" => flock_prover::pcs::ligerito::LigeritoProfile::Slim,
            _ => return Err("expected strict fast or slim profile".into()),
        }), true, args.len() == 5, initial_k),
        Some("build-info") => { println!("{}", serde_json::json!({"counters_enabled":cfg!(feature = "counters"),"terminal_enabled":cfg!(feature = "terminal")})); Ok(()) },
        Some("field") if args.len()==1 => binary::benchmark(),
        Some("parameters") if args.len()==2 => parameters::generate(args[1].as_ref()),
        Some("init-prove") if args.len()==4 || (args.len()==5 && args[4]=="--coalesced") => init::run_inner(args[1].as_ref(), args[2].as_ref(), Some(match args[3].as_str() {
            "fast" => flock_prover::pcs::ligerito::LigeritoProfile::Fast,
            "slim" => flock_prover::pcs::ligerito::LigeritoProfile::Slim,
            _ => return Err("expected strict fast or slim profile".into()),
        }), false, args.len()==5, initial_k),
        Some("fold") if args.len()==2 => fold::run(args.get(1).ok_or("missing output directory")?.as_ref()),
        Some("fri") if args.len()==2 || (args.len()==3&&args[2]=="--prove") => fri::run(
            args.get(1).ok_or("missing output directory")?.as_ref(),
            args.get(2).is_some_and(|s| s == "--prove"),
        ),
        Some("init-census") if args.len()==3 => init::census(
            args.get(1)
                .ok_or("missing Init artifact directory")?
                .as_ref(),
            args.get(2).ok_or("missing output directory")?.as_ref(),
        ),
        #[cfg(feature = "counters")]
        Some("chain") if args.len()==4 => hash_chain::run(
            args.get(1).ok_or("missing chain length")?.parse()?,
            args.get(2).map_or("slim", String::as_str),
            args.get(3).ok_or("missing output directory")?.as_ref(),
        ),
        _ => Err("usage: init-flock field | parameters <out> | fold <out> | fri <out> [--prove] | init-census <ix-artifacts> <out> | init-prove <ix-artifacts> <out> <fast|slim> [--coalesced] [--initial-k 4|5|6] | chain <length> <fast|slim> <out> | terminal-fri <out> | terminal-init <ix-artifacts> <out> [fast|slim] [--coalesced] [--initial-k 4|5|6] (terminal feature required)".into()),
    }
}
