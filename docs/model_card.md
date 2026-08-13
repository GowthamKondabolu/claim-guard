# Model card: ClaimGuard Isolation Forest v0.1.0

## Intended use

Rank enriched synthetic healthcare claims for educational payment-integrity review demonstrations.

## Not intended for

- Fraud determinations
- Payment denial or recovery decisions
- Clinical decisions
- Provider sanctions
- Use with protected health information without an approved security and governance design

## Model

Isolation Forest with standardized numeric features and one-hot encoded claim type.

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

The model is evaluated against injected synthetic scenarios using Precision@K, Recall@K, ROC-AUC, and average precision. These metrics describe recovery of engineered test patterns only.

## Known limitations

- Synthetic distributions are not equivalent to real claims distributions.
- Injected anomaly scenarios are simplified.
- An anomaly is not evidence of fraud, waste, or abuse.
- Provider-relative features can be unstable for low-volume providers.
- The Phase 1 API requires enriched features from an upstream pipeline.
- No investigator feedback loop is implemented yet.

## Human oversight

Every flagged claim requires review by an authorized, qualified investigator. Reason codes are supporting signals, not conclusions.

