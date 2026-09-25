# Model card: `fraud-detector`

All numbers below come from one reproducible run of the pipeline in this repository:
`HISTORY_END=1767225600` (history ending 2026-01-01 00:00 UTC), default settings,
package version 1.0.0. Re-running `docker compose up` with the same settings produces
the same dataset and the same model; the trainer logs every figure below to MLflow.

## Model details

| | |
|---|---|
| Type | Binary classifier: average of an XGBoost and a LightGBM gradient-boosted tree model, followed by Platt (logit) calibration. |
| Output | Calibrated probability that a card transaction is fraudulent. |
| Inputs | The 17 behavioural features defined in the [architecture document](architecture.md#feature-engineering). |
| Serving | Registered in MLflow as native model files plus a pyfunc wrapper (models from code). The predictor loads the native files through the `champion` registry alias. |
| Decision policy | `REVIEW` when p ≥ 0.1, `BLOCK` when p ≥ 0.9, otherwise `APPROVE` (`THRESHOLD_REVIEW`, `THRESHOLD_BLOCK`). |
| Training code | [`trainer.py`](../src/fraud_detection/trainer.py) |

## Intended use

- Demonstrating an end-to-end, production-style real-time fraud-scoring system:
  streaming features, model lifecycle, serving and monitoring.
- Ranking card authorisations for manual review or blocking **in the simulation that
  ships with this repository.**

**Out of scope.** The model is trained only on synthetic data. It must not be used to
make decisions about real people or real transactions. Deploying the architecture on
real data would require retraining, a fresh evaluation, and the governance steps listed
under [limitations](#caveats-and-limitations).

## Training data

Synthetic card transactions from [`simulation.py`](../src/fraud_detection/simulation.py)
(3,000 users, 600 merchants, 45 days), replayed through the production feature code.
The generated fraud patterns (account-takeover bursts, card testing, opportunistic
card-not-present fraud) are described in the
[architecture document](architecture.md#simulation).

| Split (event time) | Period (UTC) | Rows | Fraud rate |
|---|---|---|---|
| Warm-up (dropped) | 2025-11-17 to 2025-11-27 | ~46k | – |
| Train | 2025-11-27 to 2025-12-21 | 112,277 | 0.89% |
| Validation | 2025-12-21 to 2025-12-26 | 24,059 | 0.91% |
| Test | 2025-12-26 to 2026-01-01 | 24,059 | 0.97% |

The split is **out-of-time**: the model never sees data from after its training period,
and the test period is untouched until the final evaluation. The first ten days are
dropped because rolling windows (up to 32 days) are still filling there.

## Evaluation (test period)

| Metric | Value |
|---|---|
| PR-AUC (average precision) | **0.792** (95% CI 0.707–0.853) |
| ROC-AUC | 0.987 |
| Recall at 1% false-positive rate | 0.850 |
| Brier score | 0.00374 |
| Log loss | 0.0157 |

The confidence interval comes from a cluster bootstrap over users (200 resamples):
fraud arrives in per-user bursts, so resampling rows would understate the
uncertainty. The test period holds about 230 fraudulent transactions. Treat
differences of a few hundredths of PR-AUC between runs as noise.

### Operating points

| Threshold | Action | Precision | Recall | Share of traffic flagged |
|---|---|---|---|---|
| 0.1 | `REVIEW` or `BLOCK` | 0.51 | 0.80 | 1.53% |
| 0.9 | `BLOCK` | 0.96 | 0.47 | 0.47% |

Other points on the precision-recall curve: 0.29 precision at 0.89 recall
(p ≥ 0.022), 0.71 at 0.74 (p ≥ 0.23), 0.95 at 0.50 (p ≥ 0.87). The full curve is logged
as `evaluation/precision_recall.csv`. The thresholds are a business choice
(review capacity versus losses) and can be changed without retraining.

### Calibration

Probabilities are calibrated: across deciles of predicted risk, the mean prediction and
the observed fraud rate agree (top decile: 0.095 predicted, 0.093 observed; see
`evaluation/reliability.csv`). The fitted Platt parameters (slope 0.98, intercept 0.27)
show that the averaged boosters were already close to calibrated, because they are
trained on log loss without class re-weighting. Calibration stays in the pipeline as a
safeguard that never reorders transactions.

### Design choices backed by this evaluation

| Choice | Evidence |
|---|---|
| Platt scaling instead of isotonic regression | Isotonic calibration collapsed ~24k distinct test scores into 29 plateaus, cutting PR-AUC from 0.792 to 0.771 and worsening log loss (0.0163 vs 0.0157). Platt scaling is strictly monotone. |
| Keep the two-model average | Validation PR-AUC: XGBoost 0.766, LightGBM 0.752, average 0.766. On the test period XGBoost alone scored 0.801, but the validation data cannot tell it apart from the average, so choosing it would be selection on noise. |
| Out-of-time split | A random split would place transactions from the same fraud burst in both train and test. |

### Most influential features

Share of total split gain, averaged over both boosters:

| Feature | Share |
|---|---|
| `tx_count_1h` | 0.23 |
| `merchant_fraud_rate_30d` | 0.18 |
| `seconds_since_last_tx` | 0.16 |
| `card_present` | 0.12 |
| `amount_zscore` | 0.08 |
| `is_new_country` | 0.06 |

`user_chargeback_rate` contributes almost nothing (0.006): with a realistic two-day
label delay, a user's own past fraud is rarely known in time to help.

## Monitoring

The monitor compares live decisions with this model's test-period predictions (the
reference dataset logged with the training run): per-feature and score drift, plus
precision, recall and ROC-AUC at the `REVIEW` threshold on labelled decisions. Calendar
features are excluded from drift tests because a short window cannot match their
multi-day reference distribution. See the [architecture document](architecture.md#monitoring).

## Ethical considerations

- **No real personal data.** Users, merchants and transactions are simulated.
- **No protected attributes.** Features describe behaviour: amounts, velocity, merchant
  risk and time. Country is used only *relative to the user's own history*
  (`is_new_country`, `unique_countries_7d`), and the simulator assigns fraudsters'
  countries uniformly, so the model cannot learn that any country is risky.
- **Human in the loop.** Most flagged transactions go to `REVIEW`, not an automatic
  block. On real data, block decisions would need an appeal path.
- **Real-data fairness.** Behavioural features can still correlate with protected
  characteristics in real populations (for example, travel frequency). A real
  deployment should measure error rates across relevant groups before use.

## Caveats and limitations

- **Synthetic performance only.** The metrics measure how well the model learns the
  simulator's fraud patterns. Real fraud is adversarial and changes over time.
- **Labels.** The simulator attaches ground truth to each event. Label-derived
  features respect a two-day label delay, but real chargebacks arrive later and
  unevenly.
- **Small positive class.** About 230 fraudulent transactions in the test period;
  see the confidence interval above.
- **Cold start.** Users with no history get neutral values (for example
  `amount_zscore = 0`), which lowers recall for their first transactions.
