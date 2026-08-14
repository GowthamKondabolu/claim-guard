# Model card: ClaimGuard ensemble v0.2.0

## Intended use

Rank enriched synthetic healthcare claims for educational payment-integrity review demonstrations. ClaimGuard prioritizes claims for qualified human review; it does not determine fraud, waste, or abuse.

## Not intended for

- Fraud determinations
- Payment denial or recovery decisions
- Clinical decisions
- Provider sanctions
- Use with protected health information without an approved security and governance design

## Scoring system

### Isolation Forest

The unsupervised model uses standardized numeric features and one-hot encoded claim type. Raw Isolation Forest scores are converted to empirical percentiles using the persisted training-score reference distribution.

### Transparent rule baseline

The rule score is bounded between 0 and 1 using explicit, testable signals:

| Rule | Threshold | Rule-score weight |
|---|---:|---:|
| High provider-relative cost | Provider paid-amount z-score ≥ 3 | 0.30 |
| Excess units | Units ≥ 20 | 0.20 |
| Possible duplicate | Duplicate indicator = 1 | 0.25 |
| High provider utilization | Provider claims in 30 days ≥ 40 | 0.15 |
| High beneficiary utilization | Beneficiary claims in 30 days ≥ 12 | 0.10 |

These thresholds are illustrative engineering defaults, not clinical or payment policies.

### Ensemble

The default score combines 90% calibrated Isolation Forest percentile and 10% rule score. Training persists the score-reference distribution, ensemble weight, and review threshold so offline and API scoring use the same contract.

## Inputs

- Log paid amount
- Diagnosis and procedure counts
- Units and length of stay
- Provider and beneficiary 30-day claim counts
- Provider-relative paid-amount z-score
- Duplicate indicator
- Weekend-service indicator
- Claim type

## Evaluation

Synthetic labels are used only for evaluation and never as training features. With the committed seed, 5,000 generated claims, 200 injected scenarios, and a 200-claim review capacity:

| Ranking strategy | Precision@200 | Recall@200 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.440 | 0.440 | 0.9166 | 0.4791 |
| Transparent rules | 0.190 | 0.190 | 0.7582 | 0.1805 |
| 90/10 ML-rule ensemble | 0.460 | 0.460 | 0.9208 | 0.3896 |

The ensemble improves the committed run’s top-200 recovery and ROC-AUC, while average precision is lower than the Isolation Forest alone. The 90/10 weight was selected for this synthetic demonstration and has not been validated on an independent real-world dataset.

## Known limitations

- Synthetic distributions are not equivalent to real claims distributions.
- Injected anomaly scenarios are simplified.
- An anomaly or rule trigger is not evidence of fraud, waste, or abuse.
- Rule thresholds and weights are illustrative and require domain validation.
- The ensemble weight was inspected against the same synthetic evaluation scenario.
- Provider-relative features can be unstable for low-volume providers.
- Score calibration and thresholds can become stale as data distributions drift.
- The API requires enriched features from an upstream pipeline.
- No investigator feedback loop is implemented yet.

## Human oversight

Every flagged claim requires review by an authorized, qualified investigator. Scores and reason codes are supporting signals, not conclusions.
