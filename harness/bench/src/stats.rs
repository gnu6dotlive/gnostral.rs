//! Canonical derived statistics.
//!
//! The receipt stores every sample and the derived values. Validation recomputes
//! the derived values with exactly the formulas below, so a receipt whose
//! summary was edited away from its samples is rejected.
//!
//! * `median`: sort ascending; odd count takes the middle sample, even count
//!   takes the arithmetic mean of the two middle samples.
//! * `min` / `max`: first and last of the sorted samples.
//! * `p95`: nearest-rank, `ceil(0.95 * n)` (1-based), clamped to `n`.
//!
//! Non-finite or empty input has no defined statistic and yields `None`.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p95: f64,
}

pub fn statistics(samples: &[f64]) -> Option<Stats> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let middle = count / 2;
    let median = if count % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    };
    // ceil(0.95 * n) without floating-point rounding: ceil(95n / 100).
    let rank = (95 * count).div_ceil(100);
    let p95 = sorted[rank.saturating_sub(1).min(count - 1)];
    Some(Stats {
        median,
        min: sorted[0],
        max: sorted[count - 1],
        p95,
    })
}

#[cfg(test)]
mod tests {
    use super::statistics;

    #[test]
    fn odd_count_median_is_the_middle_sample() {
        let stats = statistics(&[40.0, 30.0, 34.0]).expect("finite samples");
        assert_eq!(stats.median, 34.0);
        assert_eq!(stats.min, 30.0);
        assert_eq!(stats.max, 40.0);
        assert_eq!(stats.p95, 40.0);
    }

    #[test]
    fn even_count_median_averages_the_two_middle_samples() {
        let stats = statistics(&[4.0, 1.0, 3.0, 2.0]).expect("finite samples");
        assert_eq!(stats.median, 2.5);
        assert_eq!(stats.p95, 4.0);
    }

    #[test]
    fn p95_uses_nearest_rank() {
        let samples: Vec<f64> = (1..=20).map(f64::from).collect();
        let stats = statistics(&samples).expect("finite samples");
        assert_eq!(stats.p95, 19.0);
        let single = statistics(&[7.0]).expect("finite sample");
        assert_eq!(single.p95, 7.0);
    }

    #[test]
    fn empty_or_non_finite_has_no_statistic() {
        assert!(statistics(&[]).is_none());
        assert!(statistics(&[1.0, f64::INFINITY]).is_none());
    }
}
