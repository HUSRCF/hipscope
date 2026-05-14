//! `hipfire-atlas` CLI — minimal entry-point for the Rust collection
//! layer. Today this binary only handles **corpus reading** (verify a
//! row, count rows, dump as pretty JSON). The collection itself happens
//! inside the bench binaries (`bench_qwen35_mq4 --emit-atlas <path>`,
//! etc.) which depend on the `hipfire-atlas` library directly.
//!
//! The Python harness `scripts/kernel_atlas.py` is the analysis layer
//! and stays in Python — pandas-style ranking iterates faster there.

use hipfire_atlas::load_rows;

fn print_usage() {
    eprintln!("hipfire-atlas — kernel atlas corpus tool");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  hipfire-atlas read <path.jsonl>          show row count + first row pretty");
    eprintln!("  hipfire-atlas count <path.jsonl>         print number of rows");
    eprintln!("  hipfire-atlas head <path.jsonl> [N]      print first N rows (default 1)");
    eprintln!();
    eprintln!("For collection: use the bench tools' --emit-atlas <path> flag.");
    eprintln!("For analysis: scripts/kernel_atlas.py on the HIPa branch.");
}

fn cmd_count(path: &str) -> Result<(), String> {
    let rows = load_rows(path)?;
    println!("{}", rows.len());
    Ok(())
}

fn cmd_head(path: &str, n: usize) -> Result<(), String> {
    let rows = load_rows(path)?;
    for row in rows.iter().take(n) {
        let pretty = serde_json::to_string_pretty(row)
            .map_err(|e| format!("serialize row: {e}"))?;
        println!("{pretty}");
    }
    Ok(())
}

fn cmd_read(path: &str) -> Result<(), String> {
    let rows = load_rows(path)?;
    println!("rows: {}", rows.len());
    if let Some(first) = rows.first() {
        let pretty = serde_json::to_string_pretty(first)
            .map_err(|e| format!("serialize first row: {e}"))?;
        println!("first row:");
        println!("{pretty}");
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Err("missing subcommand".into());
    }
    match args[1].as_str() {
        "read" if args.len() == 3 => cmd_read(&args[2]),
        "count" if args.len() == 3 => cmd_count(&args[2]),
        "head" if args.len() == 3 => cmd_head(&args[2], 1),
        "head" if args.len() == 4 => {
            let n: usize = args[3].parse().map_err(|e| format!("invalid N: {e}"))?;
            cmd_head(&args[2], n)
        }
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        other => {
            print_usage();
            Err(format!("unknown or malformed subcommand: {other}"))
        }
    }
}

fn main() {
    if let Err(msg) = run() {
        eprintln!("error: {msg}");
        std::process::exit(1);
    }
}

// Self-check: AtlasRow round-trips through JSONL without losing the
// flatten'd extra fields.
#[cfg(test)]
mod tests {
    use hipfire_atlas::AtlasRow;
    use serde_json::json;

    #[test]
    fn row_roundtrip() {
        let mut row = AtlasRow::new("ar", "qwen3.5-9b");
        row.set_metric_f64("prefill_tok_s", 1432.5)
            .set_metric_f64("prefill_kernel_ms", 88.4)
            .set_extra("git_sha", json!("46632a35"));
        let text = serde_json::to_string(&row).unwrap();
        let back: AtlasRow = serde_json::from_str(&text).unwrap();
        assert_eq!(back.workload_kind, "qwen3.5-9b");
        assert_eq!(back.metric_f64("prefill_tok_s"), Some(1432.5));
        assert_eq!(back.extra.get("git_sha"), Some(&json!("46632a35")));
    }
}
