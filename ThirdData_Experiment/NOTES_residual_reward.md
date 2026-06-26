# Side investigation: residual / boosting reward for formula search

> Exploratory finding — **not part of the main reported results** (ThirdData's
> headline stays 0.833 with the standard pipeline). Recorded here for reference.

## Motivation

Observed across datasets: a small number of formulas already reaches near-peak
accuracy, and adding many more barely helps (Brain 6000->800 no loss; ThirdData
+pair-search no gain; COVIDx 25-formula smoke 0.866 vs 800-formula 0.895). The
RL search reward is **decoupled** from the classifier's errors: each formula is
rewarded for its own standalone discriminability + a diversity (decorrelation)
penalty -- never for reducing what the current ensemble still gets wrong. Idea:
add a "residual connection" / boosting signal so new formulas target the
ensemble's residual error (functional-gradient boosting at the feature-discovery
level).

## What was implemented (prototype, in src/rl/tensor_environment_large_bank.py)

A `reward_type: residual` mode:
- A fixed reference eval batch is cached for alignment.
- Every `residual_refresh` episodes, a light linear classifier is trained on the
  current feature bank's outputs (optionally seeded with accumulated formulas
  from previous rounds via `residual_seed_path`) -> per-sample pseudo-residual
  R = onehot(y) - softmax(ensemble).
- A new formula's reward = max over classes of |corr(formula_feature, R[:,c])|
  (how much it explains the residual direction), minus the diversity penalty.
- Round-based boosting: round k is seeded with rounds 0..k-1's formulas so it
  targets what the accumulated ensemble still misses.

Configs/runs: configs/thirddata_residual.yaml (round 1),
configs/thirddata_res2.yaml (round 2, seeded), feature combination tester
scripts/combine_features_multi.py.

## Results (ThirdData / BUSI, 16-region encoding, fixed test set)

Feature-set combination (concatenate per-run features, then HGB; bypasses step-3
quality re-filtering which otherwise discards the low-global-F residual formulas):

| feature set | acc | bal_acc | normal |
|---|---|---|---|
| baseline (standard reward) | 0.825 | 0.740 | 0.455 |
| baseline + residual round-1 | 0.833 | 0.776 | 0.591 |
| baseline + residual round-2 (seeded on base+r1) | 0.833 | 0.776 | 0.591 |
| baseline + residual r1 + r2 | 0.800 | 0.735 | 0.545 |

## Findings

1. The residual reward works (one round): adding residual-rewarded formulas to
   the baseline pool lifted acc 0.825->0.833 and balanced acc 0.740->0.776, by
   rescuing the weak `normal` class (0.455->0.591). Direct evidence that
   residual-aware search finds COMPLEMENTARY formulas the standard reward does
   not -- the "residual connection" intuition is correct.
2. It must bypass step-3 selection: combining the pools and re-running the
   global-quality step-3 gate LOSES the gain (the complementary residual formulas
   have low global ANOVA-F and get filtered out). The gain only appears when the
   residual features are concatenated directly.
3. Saturates after one round on this dataset: a properly-seeded round 2 (verified
   it targets the base+round1 residual; its best reward dropped 0.549->0.243, a
   harder residual) found formulas EQUIVALENT to round 1, and stacking all three
   slightly hurt (redundancy + dimensionality). ThirdData has very little headroom
   (test 120 imgs, normal only 22), so the residual is exhausted in one round.
4. Caveat / next step: a real bug was found and fixed mid-experiment -- the
   round-2 residual ensemble had been capped to the first 500 seed formulas, which
   were all baseline (residual1 excluded); after the fix round 2 was truly seeded
   on base+round1. Multi-round scaling should be re-tested on a higher-headroom
   dataset (e.g. full COVIDx, or settings with more minority data) before
   concluding whether >1 round ever helps.

## Bottom line

Residual-aware reward is a promising, working mechanism for a SINGLE complementary
boost (esp. for a weak minority class), but did not show multi-round scaling on
ThirdData. Kept out of the headline results; worth revisiting on larger /
higher-headroom data.
