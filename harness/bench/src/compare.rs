//! Cross-lane comparison: refuse incomparable lanes, then render the Pareto
//! table. A ratio is only ever printed between two known numbers, and next to
//! it the full sample range and whether it separates from the baseline's: a
//! median ratio between overlapping ranges is not a ranking.

use crate::receipt::{DriverStatus, Parity, ParityClass, Receipt};
use crate::stats::{statistics, Stats};

/// Compare two or more already-validated receipts. Returns the Markdown report
/// on success, or the complete list of reasons the comparison was refused.
pub fn compare(lanes: &[(String, Receipt)]) -> Result<String, Vec<String>> {
    let mut refusals = Vec::new();
    let Some((base_label, base)) = lanes.first() else {
        return Err(vec!["no receipts supplied".to_string()]);
    };
    if lanes.len() < 2 {
        return Err(vec![
            "at least two receipts are required to compare".to_string()
        ]);
    }

    for (label, lane) in lanes {
        if lane.lane.parity_class == ParityClass::Exact && lane.semantic_parity != Parity::Pass {
            refusals.push(format!(
                "{label}: semantic parity is {}, and a lane that did not pass the frozen probe cannot be ranked",
                lane.semantic_parity.as_str()
            ));
        }
        if lane.self_reproducibility != Parity::Pass {
            refusals.push(format!(
                "{label}: self-reproducibility is {}, and a lane that does not reproduce itself cannot be ranked",
                lane.self_reproducibility.as_str()
            ));
        }
        if lane.lane.placement.is_none() {
            refusals.push(format!(
                "{label}: layer placement is UNKNOWN, and placement is part of a lane's identity"
            ));
        }
        if lane.lane.driver_status != DriverStatus::NoXid {
            refusals.push(format!(
                "{label}: driver status is {}, expected NO_XID",
                lane.lane.driver_status.as_str()
            ));
        }
        match lane.recomputed() {
            None => refusals.push(format!("{label}: no finite decode samples to rank")),
            Some(stats) if !stats.median.is_finite() || stats.median <= 0.0 => refusals.push(
                format!("{label}: decode median is not a finite positive number"),
            ),
            Some(_) => {}
        }

        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.id",
            &base.protocol.id,
            &lane.protocol.id,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.sha256",
            &base.protocol.sha256,
            &lane.protocol.sha256,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.context_tokens",
            &base.protocol.context_tokens,
            &lane.protocol.context_tokens,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.prompt_sha256",
            &base.protocol.prompt_sha256,
            &lane.protocol.prompt_sha256,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.threads",
            &base.protocol.threads,
            &lane.protocol.threads,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.sampling.temperature",
            &base.protocol.sampling.temperature,
            &lane.protocol.sampling.temperature,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.sampling.seed",
            &base.protocol.sampling.seed,
            &lane.protocol.sampling.seed,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.sampling.n_predict",
            &base.protocol.sampling.n_predict,
            &lane.protocol.sampling.n_predict,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "protocol.sampling.repeat_penalty",
            &base.protocol.sampling.repeat_penalty,
            &lane.protocol.sampling.repeat_penalty,
        );
        frozen(
            &mut refusals,
            label,
            base_label,
            "lane.model_sha256",
            &base.lane.model_sha256,
            &lane.lane.model_sha256,
        );
    }

    if !refusals.is_empty() {
        return Err(refusals);
    }

    Ok(render(lanes))
}

fn frozen<T: PartialEq + std::fmt::Debug>(
    refusals: &mut Vec<String>,
    label: &str,
    base_label: &str,
    field: &str,
    expected: &T,
    actual: &T,
) {
    if expected != actual {
        refusals.push(format!(
            "{label}: {field} is {actual:?}, expected {expected:?} from {base_label}"
        ));
    }
}

struct Row {
    engine: String,
    class: &'static str,
    placement: String,
    separation: &'static str,
    stats: Stats,
    sample_count: usize,
    vram_median: Option<f64>,
    ambient: Option<f64>,
    speed_ratio: f64,
    vram_ratio: Option<f64>,
    pareto: &'static str,
}

fn vram_median(receipt: &Receipt) -> Option<f64> {
    let values: Option<Vec<f64>> = receipt
        .samples
        .iter()
        .map(|sample| sample.vram_used_mib)
        .collect();
    statistics(&values?).map(|stats| stats.median)
}

fn dominates(a: &Row, b: &Row) -> bool {
    let (Some(a_vram), Some(b_vram)) = (a.vram_median, b.vram_median) else {
        return false;
    };
    let not_worse = a.stats.median >= b.stats.median && a_vram <= b_vram;
    let strictly_better = a.stats.median > b.stats.median || a_vram < b_vram;
    not_worse && strictly_better
}

fn render(lanes: &[(String, Receipt)]) -> String {
    let base = &lanes[0].1;
    let base_stats = base.recomputed().expect("validated receipts have samples");
    let baseline_median = base_stats.median;
    let baseline_vram = vram_median(base);

    let mut rows: Vec<Row> = lanes
        .iter()
        .enumerate()
        .map(|(index, (_, lane))| {
            let stats = lane.recomputed().expect("validated receipts have samples");
            let vram_median = vram_median(lane);
            let speed_ratio = stats.median / baseline_median;
            let vram_ratio = match (baseline_vram, vram_median) {
                (Some(base_vram), Some(vram)) => Some(vram / base_vram),
                _ => None,
            };
            Row {
                engine: lane.lane.engine.clone(),
                class: lane.lane.parity_class.as_str(),
                placement: lane
                    .lane
                    .placement
                    .clone()
                    .unwrap_or_else(|| "UNKNOWN".into()),
                separation: if index == 0 {
                    "baseline"
                } else {
                    separation(&stats, &base_stats)
                },
                stats,
                sample_count: lane.samples.len(),
                vram_median,
                ambient: lane.ambient_vram_mib,
                speed_ratio,
                vram_ratio,
                pareto: "UNKNOWN",
            }
        })
        .collect();

    let all_vram_known = rows.iter().all(|row| row.vram_median.is_some());
    if all_vram_known {
        for index in 0..rows.len() {
            let dominated = rows
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != index && dominates(candidate, &rows[index]));
            rows[index].pareto = if dominated { "DOMINATED" } else { "FRONT" };
        }
    }

    let mut out = String::new();
    out.push_str("Baseline (first lane): ");
    out.push_str(&cell(&base.lane.engine));
    out.push('\n');
    for (label, lane) in lanes {
        out.push_str(&format!(
            "{}: median {} tok/s over {} samples (receipt {})\n",
            cell(&lane.lane.engine),
            format_3(lane.recomputed().map(|stats| stats.median).unwrap_or(0.0)),
            lane.samples.len(),
            cell(label)
        ));
    }
    out.push('\n');
    out.push_str("| Lane | Class | Placement | Decode tok/s (median) | Decode min-max | n | vs baseline range | Whole-card VRAM MiB (median) | Ambient VRAM MiB | Speed vs baseline | VRAM vs baseline | Pareto |\n");
    out.push_str(
        "| --- | --- | --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | :---: |\n",
    );
    for row in &rows {
        let vram = row
            .vram_median
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let ambient = row
            .ambient
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let vram_ratio = row
            .vram_ratio
            .map(|value| format!("{value:.3}x"))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        out.push_str(&format!(
            "| {} | {} | {} | {} | {}-{} | {} | {} | {} | {} | {:.3}x | {} | {} |\n",
            cell(&row.engine),
            row.class,
            cell(&row.placement),
            format_3(row.stats.median),
            format_3(row.stats.min),
            format_3(row.stats.max),
            row.sample_count,
            row.separation,
            vram,
            ambient,
            row.speed_ratio,
            vram_ratio,
            row.pareto,
        ));
    }
    if rows
        .iter()
        .any(|row| row.class == ParityClass::SelfOnly.as_str())
    {
        out.push_str("\nSelf-class lanes run a different engine: they are gated on reproducing their own output, not on token identity with the baseline. Their agreement with the reference is in their run output, not ranked here.\n");
    }
    out
}

/// Whether `lane`'s sample range lies entirely above or below the baseline's.
fn separation(lane: &Stats, base: &Stats) -> &'static str {
    if lane.min > base.max {
        "ABOVE"
    } else if lane.max < base.min {
        "BELOW"
    } else {
        "OVERLAP"
    }
}

fn format_3(value: f64) -> String {
    format!("{value:.3}")
}

/// Keep a Markdown table cell from breaking on a pipe or newline.
fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::{dominates, Row};
    use crate::stats::Stats;

    fn row(median: f64, vram: Option<f64>) -> Row {
        Row {
            engine: "lane".into(),
            class: "exact",
            placement: "0-0:cpu;nm=cpu".into(),
            separation: "OVERLAP",
            stats: Stats {
                median,
                min: median,
                max: median,
                p95: median,
            },
            sample_count: 3,
            vram_median: vram,
            ambient: None,
            speed_ratio: 1.0,
            vram_ratio: None,
            pareto: "UNKNOWN",
        }
    }

    #[test]
    fn domination_requires_no_worse_on_both_axes() {
        assert!(dominates(&row(10.0, Some(1000.0)), &row(5.0, Some(2000.0))));
        assert!(!dominates(
            &row(10.0, Some(3000.0)),
            &row(5.0, Some(2000.0))
        ));
        assert!(!dominates(
            &row(10.0, Some(2000.0)),
            &row(10.0, Some(2000.0))
        ));
        assert!(!dominates(&row(10.0, None), &row(5.0, Some(2000.0))));
    }
}
