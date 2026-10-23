//! `gnostral-bench` — Q-031 `ENGINE-PERF-01` receipt gate.
//!
//! This binary validates synthetic or real receipts and compares pinned lanes.
//! It never runs an engine, never touches a GPU and never fills in a missing
//! measurement.

use std::fs;

use gnostral_bench::compare;
use gnostral_bench::json;
use gnostral_bench::receipt::{self, is_sha256_hex, Receipt};
use gnostral_bench::sha256;

const USAGE: &str = "\
usage:
  gnostral-bench validate <receipt.json> [--protocol-hash <sha256> | --protocol-file <path>]
  gnostral-bench compare <a.json> <b.json> [<c.json> ...] [--protocol-hash <sha256> | --protocol-file <path>]
  gnostral-bench --help

exit codes: 0 valid / compared, 1 invalid receipt or refused comparison, 2 usage or I/O error";

fn main() {
    std::process::exit(run(std::env::args().skip(1).collect()));
}

fn run(args: Vec<String>) -> i32 {
    match args.first().map(String::as_str) {
        Some("validate") => cmd_validate(&args[1..]),
        Some("compare") => cmd_compare(&args[1..]),
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
