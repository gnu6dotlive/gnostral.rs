//! `gnostral-bench` — Q-031 `ENGINE-PERF-01` receipt gate.
//!
//! This binary validates synthetic or real receipts, compares pinned lanes and,
//! with `run`, drives a frozen-protocol plan. It never touches a GPU itself and
//! never fills in a missing measurement.

use std::fs;

use gnostral_bench::compare;
use gnostral_bench::json;
use gnostral_bench::receipt::{self, is_sha256_hex, Receipt};
use gnostral_bench::run;
use gnostral_bench::sha256;

const USAGE: &str = "\
usage:
  gnostral-bench validate <receipt.json> [--protocol-hash <sha256> | --protocol-file <path>]
  gnostral-bench compare <a.json> <b.json> [<c.json> ...] [--protocol-hash <sha256> | --protocol-file <path>]
  gnostral-bench run <plan.json> --out <dir> [--dry-run]
  gnostral-bench --help

exit codes: 0 valid / compared, 1 invalid receipt or refused comparison, 2 usage or I/O error";

fn main() {
    std::process::exit(run(std::env::args().skip(1).collect()));
}

fn run(args: Vec<String>) -> i32 {
    match args.first().map(String::as_str) {
        Some("validate") => cmd_validate(&args[1..]),
        Some("compare") => cmd_compare(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            0
        }
        Some(other) => {
            eprintln!("gnostral-bench: unknown command {other:?}");
            eprintln!("{USAGE}");
            2
        }
        None => {
            eprintln!("{USAGE}");
            2
        }
    }
}

#[derive(Default)]
struct Options {
    files: Vec<String>,
    protocol_hash: Option<String>,
    protocol_file: Option<String>,
    help: bool,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        match arg.as_str() {
            "-h" | "--help" => options.help = true,
            "--protocol-hash" | "--protocol-file" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(format!("{arg} requires a value"));
                };
                index += 1;
                if arg == "--protocol-hash" {
                    if options.protocol_hash.replace(value.clone()).is_some() {
                        return Err("--protocol-hash was given more than once".into());
                    }
                } else if options.protocol_file.replace(value.clone()).is_some() {
                    return Err("--protocol-file was given more than once".into());
                }
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg:?}")),
            _ => options.files.push(arg.clone()),
        }
        index += 1;
    }
    Ok(options)
}

fn expected_protocol(options: &Options) -> Result<Option<String>, String> {
    match (&options.protocol_hash, &options.protocol_file) {
        (Some(_), Some(_)) => Err("use either --protocol-hash or --protocol-file, not both".into()),
        (Some(hash), None) => {
            let hash = hash.to_ascii_lowercase();
            if !is_sha256_hex(&hash) {
                return Err(format!(
                    "--protocol-hash expects 64 hexadecimal characters, found {hash:?}"
                ));
            }
            Ok(Some(hash))
        }
        (None, Some(path)) => {
            let bytes = fs::read(path)
                .map_err(|error| format!("cannot read protocol file {path:?}: {error}"))?;
            Ok(Some(sha256::hex(&sha256::sha256(&bytes))))
        }
        (None, None) => Ok(None),
    }
}

fn load(path: &str) -> Result<Receipt, Vec<String>> {
    let text =
        fs::read_to_string(path).map_err(|error| vec![format!("cannot read file: {error}")])?;
    let value = json::parse(&text).map_err(|error| vec![error.to_string()])?;
    receipt::decode(&value).map_err(|error| vec![error])
}

fn print_issues(prefix: &str, issues: &[String]) {
    eprintln!("{prefix}");
    for issue in issues {
        eprintln!("  - {issue}");
    }
}

fn cmd_validate(args: &[String]) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("gnostral-bench: {message}");
            eprintln!("{USAGE}");
            return 2;
        }
    };
    if options.help {
        println!("{USAGE}");
        return 0;
    }
    if options.files.len() != 1 {
        eprintln!("gnostral-bench: validate takes exactly one receipt");
        eprintln!("{USAGE}");
        return 2;
    }
    let expected = match expected_protocol(&options) {
        Ok(expected) => expected,
        Err(message) => {
            eprintln!("gnostral-bench: {message}");
            return 2;
        }
    };
    let path = &options.files[0];
    let receipt = match load(path) {
        Ok(receipt) => receipt,
        Err(issues) => {
            print_issues(
                &format!("INVALID {path}: receipt could not be decoded"),
                &issues,
            );
            return 1;
        }
    };
    let issues = receipt::validate(&receipt, expected.as_deref());
    if !issues.is_empty() {
        print_issues(
            &format!("INVALID {path}: receipt failed validation"),
            &issues,
        );
        return 1;
    }
    let stats = receipt
        .recomputed()
        .expect("validated receipt has finite samples");
    println!(
        "VALID {path}: lane={:?} samples={} median_decode_tok_s={:.3}",
        receipt.lane.engine,
        receipt.samples.len(),
        stats.median
    );
    0
}

fn cmd_compare(args: &[String]) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("gnostral-bench: {message}");
            eprintln!("{USAGE}");
            return 2;
        }
    };
    if options.help {
        println!("{USAGE}");
        return 0;
    }
    if options.files.len() < 2 {
        eprintln!("gnostral-bench: compare takes at least two receipts");
        eprintln!("{USAGE}");
        return 2;
    }
    let expected = match expected_protocol(&options) {
        Ok(expected) => expected,
        Err(message) => {
            eprintln!("gnostral-bench: {message}");
            return 2;
        }
    };

    let mut lanes = Vec::new();
    let mut refused = false;
    for path in &options.files {
        let receipt = match load(path) {
            Ok(receipt) => receipt,
            Err(issues) => {
                print_issues(
                    &format!("REFUSED {path}: receipt could not be decoded"),
                    &issues,
                );
                refused = true;
                continue;
            }
        };
        let issues = receipt::validate(&receipt, expected.as_deref());
        if !issues.is_empty() {
            print_issues(
                &format!("REFUSED {path}: receipt failed validation"),
                &issues,
            );
            refused = true;
            continue;
        }
        lanes.push((path.clone(), receipt));
    }
    if refused {
        return 1;
    }

    match compare(&lanes) {
        Ok(report) => {
            print!("{report}");
            0
        }
        Err(reasons) => {
            print_issues("REFUSED: lanes are not comparable", &reasons);
            1
        }
    }
}

fn cmd_run(args: &[String]) -> i32 {
    let mut plan_path = None;
    let mut out_dir = None;
    let mut dry_run = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                let Some(dir) = args.get(index + 1) else {
                    eprintln!("gnostral-bench: --out requires a directory");
                    return 2;
                };
                out_dir = Some(dir.clone());
                index += 1;
            }
            "--dry-run" => dry_run = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return 0;
            }
            other if other.starts_with('-') => {
                eprintln!("gnostral-bench: unknown option {other:?}");
                return 2;
            }
            other => {
                if plan_path.replace(other.to_string()).is_some() {
                    eprintln!("gnostral-bench: run takes exactly one plan");
                    return 2;
                }
            }
        }
        index += 1;
    }
    let (Some(plan_path), Some(out_dir)) = (plan_path, out_dir) else {
        eprintln!("{USAGE}");
        return 2;
    };
    let plan = match fs::read_to_string(&plan_path)
        .map_err(|e| format!("cannot read plan: {e}"))
        .and_then(|t| json::parse(&t).map_err(|e| e.to_string()))
        .and_then(|v| run::decode_plan(&v))
    {
        Ok(plan) => plan,
        Err(message) => {
            eprintln!("gnostral-bench: {message}");
            return 1;
        }
    };
    let mut exec = run::ProcessExecutor;
    if dry_run {
        let mut runnable = Vec::new();
        for lane in &plan.lanes {
            let pin = run::pin_lane(lane, &mut exec);
            println!("{}: {pin:?}", lane.engine);
            if pin == run::Pin::Pinned {
                runnable.push(lane.engine.clone());
            }
        }
        for (lane, phase, i) in run::frozen_order(&plan, &runnable) {
            println!("{phase:?} {i} {lane}");
        }
        return 0;
    }
    let bundle = run::execute(&plan, &mut exec);
    let write = |name: &str, text: &str| {
        let path = std::path::Path::new(&out_dir).join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
    };
    let mut result = write("bundle.json", &json::write(&run::bundle_json(&bundle)));
    for (engine, built) in &bundle.receipts {
        if let Ok(r) = built {
            result = result.and(write(
                &format!("receipts/{engine}.json"),
                &json::write(&receipt::encode(r)),
            ));
        }
    }
    let report = match &bundle.comparison {
        Ok(table) => table.clone(),
        Err(reasons) => format!(
            "comparison refused:\n{}\n",
            reasons
                .iter()
                .map(|r| format!("- {r}"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    };
    result = result.and(write("pareto.md", &report));
    if let Err(e) = result {
        eprintln!("gnostral-bench: {e}");
        return 2;
    }
    print!("{report}");
    if bundle.comparison.is_ok() {
        0
    } else {
        1
    }
}
