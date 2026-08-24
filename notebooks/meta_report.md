# Phase 5 — Meta-Learner (XGBoost fusion)

Random seed 42. Model `meta-learner-xgboost-ieee-cis-20260824t145659z`.

## Held-out test

```
### meta-learner (XGBoost fusion, Platt-scaled) — held-out test (n=88,581, positives=3,083, base rate=3.4804%)
Threshold: 0.603522  (chosen on validation by minimising estimated cost on V-late, raised to respect a 1.0% review-capacity cap)

PR-AUC     0.4954  (95% CI 0.4791-0.5141)
           no-skill floor 0.0348 (14.2x lift)
Precision  0.8282      Recall  0.2549      F1  0.3899

Confusion matrix        Predicted
                     neg        pos
Actual  neg     85,335        163
        pos      2,297        786

False-positive cost estimate: 4,711.48 per 1,000 transactions  [IEEE-CIS amount units (consistent with USD)]
  163 false positives x 3.01 = 490.29
  2,297 false negatives (amount + 15.04) = 416,857.38
  total = 417,347.67 over 88,581 transactions
  flag rate = 1.07% (949 sent for review)
  Assumptions:
    - A false positive costs 3.01 IEEE-CIS amount units (consistent with USD) - analyst time on a manual review, or the lost margin and friction of declining a good customer.
    - A false negative costs the transaction amount plus a flat 15.04 - the chargeback fee and internal handling on top of the value lost.
    - Cost is linear in the number of mistakes: no queue-congestion effect, no customer-churn effect from repeated false declines, no recovery on disputed fraud. All three would raise the true cost of a false positive.
    - Every fraud is assumed to charge back. In practice some share is never disputed, which would lower false-negative cost.
    - Review capacity is unbounded. This is the assumption that bites hardest: the cost-minimising threshold below flags whatever share of traffic the arithmetic favours, with no ceiling on the queue it implies.

Retained blocks: engineered, tier1.

Every selection was made on validation: the retained feature blocks on V-arb, the calibrator and both thresholds on V-late, the early-stopping round on V-fit. Test selected nothing. It was, however, read more than once and the honest phrasing is not 'scored exactly once': the isotonic baseline is a second scoring, the matched-flag-rate table is a third, and the run was repeated after a score-quantisation bug was found. That repeat moved the meta-learner by 0.00002 and corrected the baseline by 0.0074, i.e. it made the shipped result less flattering, not more.
```

### Tier-1 alone, same test split

```
### Tier-1 alone (tier1-anomaly-lightgbm-ieee-cis-20260822t185154z) — held-out test (n=88,581, positives=3,083, base rate=3.4804%)
Threshold: 0.813710  (chosen on validation by flagging at most 1.0% of VALIDATION traffic, transferred to test unchanged -- the same discipline the meta-learner threshold is held to)

PR-AUC     0.5276  (95% CI 0.5117-0.5462)
           no-skill floor 0.0348 (15.2x lift)
Precision  0.8734      Recall  0.2462      F1  0.3841

Confusion matrix        Predicted
                     neg        pos
Actual  neg     85,388        110
        pos      2,324        759

False-positive cost estimate: 4,923.54 per 1,000 transactions  [IEEE-CIS amount units (consistent with USD)]
  110 false positives x 3.01 = 330.87
  2,324 false negatives (amount + 15.04) = 435,801.60
  total = 436,132.47 over 88,581 transactions
  flag rate = 0.98% (869 sent for review)
  Assumptions:
    - A false positive costs 3.01 IEEE-CIS amount units (consistent with USD) - analyst time on a manual review, or the lost margin and friction of declining a good customer.
    - A false negative costs the transaction amount plus a flat 15.04 - the chargeback fee and internal handling on top of the value lost.
    - Cost is linear in the number of mistakes: no queue-congestion effect, no customer-churn effect from repeated false declines, no recovery on disputed fraud. All three would raise the true cost of a false positive.
    - Every fraud is assumed to charge back. In practice some share is never disputed, which would lower false-negative cost.
    - Review capacity is unbounded. This is the assumption that bites hardest: the cost-minimising threshold below flags whatever share of traffic the arithmetic favours, with no ceiling on the queue it implies.
```

**Meta-learner minus Tier-1 alone: -0.0322, 95% CI [-0.0373, -0.0273].**

### Both models at a matched 1% flag rate

The two operating points above flag different shares of traffic, and precision falls as flag rate rises, so reading their precisions side by side compares two different things. This table holds the flag rate fixed. **Both cuts are quantiles of the test score vectors**, which is what makes the flag rates comparable. It is computed after the shipped thresholds were already fixed on validation, and it selects nothing.

| model | flag rate | precision | recall (count) | recall (value) | cost per 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| meta | 0.94% | 0.8257 | 0.2228 | 0.1689 | 4,817.97 |
| tier1 | 1.00% | 0.8646 | 0.2485 | 0.1500 | 4,903.53 |

## The ablation — which layers paid for themselves

### Verdicts

- engineered: retained (base block, exempt from retirement).
- tier1: retained (base block, exempt from retirement).
- tier2: retired for want of evidence at n=31,003 (1,131 positives) -- best add_one delta +0.0012, 95% CI [-0.0011, +0.0035], width 0.0046. The interval does not exclude zero, which is an absence of evidence that this layer helps, not evidence that it does not.
- tier3_served: retired for want of evidence at n=31,003 (1,131 positives) -- best leave_one_out delta -0.0001, 95% CI [-0.0030, +0.0032], width 0.0062. The interval does not exclude zero, which is an absence of evidence that this layer helps, not evidence that it does not.
- tier3_topology: retired -- leave_one_out delta -0.0061, 95% CI [-0.0100, -0.0026], which excludes zero on the NEGATIVE side at n=31,003 (1,131 positives). This block measurably degrades the model; removing it is an improvement, not merely a simplification.

### Paired deltas, measured on the V-arb validation slice

| block | direction | delta | 95% CI | width | verdict |
| --- | --- | ---: | --- | ---: | --- |
| engineered | leave_one_out | +0.0048 | [-0.0041, +0.0137] | 0.0178 | no evidence |
| tier1 | leave_one_out | +0.4263 | [+0.3969, +0.4535] | 0.0566 | keeps |
| tier2 | leave_one_out | +0.0020 | [-0.0012, +0.0049] | 0.0061 | no evidence |
| tier2 | add_one | +0.0012 | [-0.0011, +0.0035] | 0.0046 | no evidence |
| tier3_served | leave_one_out | -0.0001 | [-0.0030, +0.0032] | 0.0062 | no evidence |
| tier3_served | add_one | +0.0008 | [-0.0034, +0.0051] | 0.0084 | no evidence |
| tier3_topology | leave_one_out | -0.0061 | [-0.0100, -0.0026] | 0.0074 | no evidence |
| tier3_topology | add_one | -0.0047 | [-0.0103, +0.0010] | 0.0112 | no evidence |

Arbiter slice: 31,003 rows, 1,131 positives. Interval widths above are the honest read on what this comparison could have detected. A retirement here is an absence of evidence at this sample size, not evidence the layer is worthless.

### The same leave-one-out deltas, fitted on the train (out-of-fold) split

Reported so the two columns can be compared. Tier-2 is contaminated on train by an account-level, label-selected eligibility filter, so a disagreement between this table and the one above is a measurement of that contamination.

| block | direction | delta | 95% CI |
| --- | --- | ---: | --- |
| engineered | leave_one_out | +0.0038 | [-0.0055, +0.0136] |
| tier1 | leave_one_out | +0.3759 | [+0.3483, +0.4032] |
| tier2 | leave_one_out | -0.0050 | [-0.0084, -0.0014] |
| tier2 | add_one | -0.0010 | [-0.0039, +0.0020] |
| tier3_served | leave_one_out | -0.0035 | [-0.0070, -0.0001] |
| tier3_served | add_one | +0.0132 | [+0.0091, +0.0175] |
| tier3_topology | leave_one_out | -0.0066 | [-0.0127, -0.0002] |
| tier3_topology | add_one | +0.0064 | [-0.0006, +0.0126] |

## Tier-2 memorisation diagnostic

The autoencoder was fitted on fraud-free train windows, excluded **by account**. Clean rows belonging to fraud-touching accounts were therefore withheld from the fit for a reason unrelated to their own label, which makes them a control group that holds the label constant and varies only fit membership.

| quantity | value |
| --- | ---: |
| included_clean_rows | 35899 |
| excluded_clean_rows | 28569 |
| fraud_rows | 5856 |
| memorisation_auc | 0.950091 |
| fraud_auc | 0.862818 |
| residual_auc | 0.364853 |

- Fit membership alone moves Tier-2's error by AUC 0.9501 (0.5 = no effect), measured on 28,569 clean train rows withheld from the fit against 35,899 clean rows inside it. Both groups carry the same label, so this is memorisation, not detection.
- The naive train fraud/clean gap is AUC 0.8628. Holding fit membership constant it falls to 0.3649, so roughly +0.4980 of the apparent gap is an artefact of the eligibility filter and will not reproduce on test.

**Read the residual figure carefully: 0.3649 is below 0.5.** With fit membership held constant, Tier-2 scores fraud as *less* anomalous than clean traffic. Its apparent train-split discrimination is not weak signal - it is the autoencoder recognising rows it was fitted on, and underneath that the signal points the wrong way.

## Calibration

Brier 0.0233, expected calibration error 0.0037. Curve: `meta_calibration_curve.png`.

| bin | count | mean predicted | observed frequency |
| --- | ---: | ---: | ---: |
| [0.0, 0.1) | 83918 | 0.0141 | 0.0151 |
| [0.1, 0.2) | 1674 | 0.1543 | 0.1553 |
| [0.2, 0.3) | 1043 | 0.2867 | 0.3212 |
| [0.3, 0.4) | 349 | 0.3650 | 0.4699 |
| [0.4, 0.5) | 358 | 0.4604 | 0.4302 |
| [0.5, 0.6) | 290 | 0.5820 | 0.3966 |
| [0.6, 0.7) | 170 | 0.6333 | 0.7765 |
| [0.7, 0.8) | 119 | 0.7448 | 0.5378 |
| [0.8, 0.9) | 14 | 0.8276 | 0.7857 |
| [0.9, 1.0) | 646 | 0.9828 | 0.8963 |

### Isotonic, the alternative that was not shipped

Not shipped, and 'lost' overstates it. Isotonic is piecewise-constant, so it ties scores and moves PR-AUC by an artefact of step width; the sigmoid is strictly monotone and leaves the ranking exactly intact, which is why the sigmoid ships when PR-AUC is the headline. On calibration itself isotonic is the better of the two, and by how much is worth reading rather than waving at: compare the two ECE figures below. A large gap is not a free win for isotonic -- it is a signal worth investigating, because a correctly-fitted sigmoid on this data should be close.

- isotonic: {'pr_auc': 0.48235, 'brier': 0.022905, 'expected_calibration_error': 0.002178, 'distinct_scores': 25}
- sigmoid: {'pr_auc': 0.495416, 'brier': 0.023298, 'expected_calibration_error': 0.003682, 'distinct_scores': 105}

## Operating points

- Review at `6.035219e-01` — minimising estimated cost on V-late, raised to respect a 1.0% review-capacity cap.
- Block at `6.052732e-01` — lowest V-late threshold above the review point reaching precision 80% over at least 50 flagged rows.

Thresholds are printed in scientific notation because an earlier build of this layer emitted probabilities many orders of magnitude below 1, which fixed-point formatting rendered as `0.000000`. That was a calibrator fitted on one scale and applied to another, not a property of the data; it is fixed, and the notation is kept only so a recurrence stays visible.

```
False-positive cost estimate: 3,357.69 per 1,000 transactions  [IEEE-CIS amount units (consistent with USD)]
  42 false positives x 3.01 = 126.33
  510 false negatives (amount + 15.04) = 74,233.06
  total = 74,359.40 over 22,146 transactions
  flag rate = 0.96% (212 sent for review)
  Assumptions:
    - A false positive costs 3.01 IEEE-CIS amount units (consistent with USD) - analyst time on a manual review, or the lost margin and friction of declining a good customer.
    - A false negative costs the transaction amount plus a flat 15.04 - the chargeback fee and internal handling on top of the value lost.
    - Cost is linear in the number of mistakes: no queue-congestion effect, no customer-churn effect from repeated false declines, no recovery on disputed fraud. All three would raise the true cost of a false positive.
    - Every fraud is assumed to charge back. In practice some share is never disputed, which would lower false-negative cost.
    - Review capacity is unbounded. This is the assumption that bites hardest: the cost-minimising threshold below flags whatever share of traffic the arithmetic favours, with no ceiling on the queue it implies.
```

```
Cost sensitivity (both parameters scaled together)
  factor   threshold        total cost        FP        FN   flag rate
    0.5x      0.011951         16,438.14     6,412        61     31.75%
    1.0x      0.015629         24,820.13     4,069       103     20.98%
    1.5x      0.018735         30,536.76     2,863       134     15.39%
```

```
Review cost sensitivity (false-positive cost alone)
  factor   threshold        total cost        FP        FN   flag rate
    1.0x      0.015629         24,820.13     4,069       103     20.98%
    5.0x      0.052884         47,594.44     1,141       237      7.15%
   25.0x      0.359459         75,129.77       150       427      1.82%
  100.0x      0.603522         86,866.23        42       510      0.96%
```

## Measurement design

### Validation, cut three ways

| slice | rows | positives | base rate | first event | last event |
| --- | ---: | ---: | ---: | --- | --- |
| V-fit | 35,432 | 1,231 | 3.4743% | 2018-03-31 19:26:43+00:00 | 2018-04-12 12:31:11+00:00 |
| V-arb | 31,003 | 1,131 | 3.6480% | 2018-04-12 12:31:34+00:00 | 2018-04-24 02:31:07+00:00 |
| V-late | 22,146 | 680 | 3.0705% | 2018-04-24 02:31:25+00:00 | 2018-05-02 05:17:20+00:00 |

### The out-of-fold handicap

Fold models are fitted on less data than the model that serves, so the out-of-fold Tier-1 column is noisier than the column the meta-learner meets at test. Measured on the untouched validation split rather than argued away:

| fold | train rows | train positives | validation PR-AUC |
| --- | ---: | ---: | ---: |
| 2 | 82,675 | 2,221 | 0.4200 |
| 3 | 165,351 | 4,571 | 0.4587 |
| 4 | 248,026 | 8,063 | 0.4776 |
| 5 | 330,702 | 11,180 | 0.5202 |
| full | 413,378 | 14,538 | 0.6155 |

The bias this introduces is conservative in direction: a noisier Tier-1 column makes the meta-learner rely on it *less* than it should, so the scheme should cost PR-AUC rather than manufacture it. That is an argument, not a measurement - it is not independently verified here, and it is load-bearing in the conclusion.

## Limitations

- **The shipped booster early-stopped at iteration 2**, and scoring uses exactly that many trees. A model whose validation metric peaks that early has very little to learn beyond its strongest input, which is consistent with the ablation - but it means the loss against Tier-1 is **confounded** between the out-of-fold handicap and simply not fitting. Separating them needs a diagnostic run fitted on the in-sample Tier-1 column, which was not performed.
- **The two ablation columns differ in fit size as well as in contamination** (330,703 rows against 35,432), so their disagreement is not cleanly attributable to either.
- **Tier-3's ring scorer is in-sample on train**, with no out-of-fold remedy. It does not reach the shipped model, which retains no Tier-3 column, but it is the likely reason the train-fitted column rates Tier-3 far higher than the arbiter does.
- **The aggregate calibration figure flatters the band the threshold sits in.** Expected calibration error is 0.0037 against a base rate of 3.4804%, but that average is dominated by the near-zero bin holding the overwhelming majority of rows. In the sparse high-probability bands where the operating threshold actually falls, predicted and observed diverge materially -- read the per-bin table above, not the summary.
- **The shipped threshold overshoots the capacity cap it is named after.** It is described as respecting a 1% review cap, chosen on V-late, but realises 1.07% on test. The cap binds on the split it was chosen on; transferring a threshold across a base-rate shift does not preserve the flag rate, and nothing re-checks it downstream.
- **Latency was not benchmarked**; the registry entry carries an empty latency block.
- **No false-negative profiling code exists for this layer**, unlike Tiers 2 and 3. The observed-failure analysis in `app/models/README.md` was computed by hand and does not regenerate with this report.

