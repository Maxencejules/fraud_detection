# Evaluation and reproducibility contract

The evaluation functions reject nonfinite event times, invalid split fractions,
nonbinary labels, and invalid probability vectors. Repeated `transaction_id` values
are rejected when the identifier column is present: replayed rows must not appear
on both sides of an evaluation boundary. Training still expects the existing
17-feature contract; identifiers and labels are not model inputs.

Fractions target row counts after warm-up. Each boundary moves to the beginning of
its timestamp group, assigning that entire group to the later period. Therefore
`max(train time) < min(validation time)` and
`max(validation time) < min(test time)`; tied timestamps never straddle periods. Every period
must contain both classes. Coarse timestamps can change the achieved fractions or
make the requested split impossible, in which case training fails explicitly.

Average precision, ROC-AUC, Brier error and decisions use the original valid
probabilities. A fixed clipping floor would create artificial ties and can change
both ranking and decisions. Log loss handles endpoint clipping internally; finite
logit fitting uses machine precision only. Serving calibration treats 0 and 1 as
the limits of the increasing logit transform, and identity calibration preserves
input scores exactly. Extreme calibrated outputs can still round to 0/1 in float64.
See the primary definitions of
[average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html),
and [sigmoid calibration](https://scikit-learn.org/stable/modules/calibration.html).

Uncertainty uses a percentile bootstrap over whole users, preserving their complete
transaction histories even when a user is sampled repeatedly. The trainer reports
requested replicates, valid two-class replicates and user count. Single-class
replicates are omitted, so the interval is conditional on both classes being
present; fewer than two valid replicates produces an error. This interval does not
measure simulator realism, time-regime shift or bias from repeated model selection.

The validation period supplies both early stopping and calibration. The later
evaluation period supplies reported metrics **and the promotion comparison**. It is
held out from fitting, but repeated promotion decisions on that same period make it
a selection set. An independent future period is needed for a final generalization
claim after model choices are frozen. The current system remains a synthetic demo.

## Bounded offline evidence

```bash
uv sync --locked --all-extras
uv run python scripts/offline_validation.py --out reports/offline-validation
```

This command starts no services and uses no external dataset or tracking server.
It generates 28 days for 400 simulated users and 120 merchants with seed 42 and a
fixed history end. Five warm-up days keep this check small; they do not fully fill
the longest feature window. Its metrics characterize this small fixture, not the
archived 45-day experiment or real fraud.

It trains with one worker thread, retrains after flipping only evaluation labels,
and requires identical predictions and fitted calibration parameters. It also
reloads the actual native XGBoost/LightGBM files and requires maximum absolute
prediction difference at most `1e-12`. Artifacts contain the generated features,
evaluation predictions, native files, split/fit/metric details, bootstrap counts,
environment versions and SHA256 fingerprints. The dataset hash uses UTF-8/LF CSV
with 17 significant digits; source hashes normalize CRLF to LF.

CI uploads this evidence while retaining the strict typing, 85% coverage, both
Python versions, dependency audit and complete Compose end-to-end gates. Exact
repeatability is checked within the same pinned environment; predictions need not
be bit-identical across operating systems or future library versions.

A [recorded Windows report](results/offline-validation-windows.json) retains the
measured small-fixture result and its source/environment fingerprints. It was
generated from the local dirty workspace before committing these source changes;
its annotation identifies the baseline and raw-report hash, not a future commit
or GitHub CI result. Two separate
executions produced six byte-identical files (the report, dataset, predictions and
three native model files). The evaluation period has only ten positives, and its
wide PR-AUC interval must accompany the point estimate. Re-run the command to obtain
the full raw artifacts; they are generated locally and uploaded by CI.
