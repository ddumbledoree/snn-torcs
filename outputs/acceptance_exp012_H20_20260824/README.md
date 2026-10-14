# exp012 H20 Acceptance Package

## Weights

```text
ours SNN T=4:
models/exp012_ours_t4_best.pth
sha256: 61c9982fed215cb04f516a0909edd02acc312cdd4a25a8c80090c42f2d7b5e25

baseline SNN T=8:
models/exp013_baseline_t8_best.pth
sha256: 22825d7b1a1455c9b67c86f5bae06000148a0328600223119cb2936fe6e9dac0
```

## Main Acceptance

Metric 1: decision speed

```text
L_base = mean forward latency of baseline SNN(T=8)
L_ours = mean forward latency of exp012 ours SNN(T=4)
speed_gain = (L_base - L_ours) / L_base
pass iff speed_gain >= 10%

Measured:
baseline mean = 1.0565 ms
ours mean     = 0.6067 ms
speed_gain    = 42.58%
Result        = pass
```

Metric 2: H20 decoded-control validity, standard condition

```text
condition = baseline_t0.15
track     = g-track-1
car       = car1-trb1
accel     = 0.15
grip      = stock
H         = 20

valid / total = 905 / 1000
H20           = 90.50%
Result        = pass
```

## Robustness Evidence

Complete 1000-step H20 pass conditions:

```text
baseline_t0.10  H20=95.30%
baseline_t0.15  H20=90.50%
baseline_t0.20  H20=97.70%
baseline_t0.25  H20=93.00%
```

Six-condition average, adding wet-all and alt-car:

```text
baseline_t0.10  95.30%
baseline_t0.15  90.50%
baseline_t0.20  97.70%
baseline_t0.25  93.00%
wet-all_t0.15   88.90%
alt-car_t0.15   89.30%

mean H20 = 92.45%
```

Alt-track note:

```text
alt-track_t0.15 full 1000-step run hit TORCS CommTimeout near step 556.
Last successful state before timeout:
speedX=47.2, trackPos=+0.438, angle=+0.032, distRaced=882.8

Pre-timeout saved trace:
steps = 556
H20   = 93.71%
```

Do not report alt-track as a full 1000-step pass. It is supporting evidence
that the policy was not visibly failing before the simulator stopped responding.

## Files

```text
reports/metric1_latency_benchmark.json
reports/standard/baseline_t0.15_steptrace.json
reports/standard/baseline_t0.15_H20.json
reports/condition_matrix/summary_H20_6plus.csv
reports/condition_matrix/summary_H20_acceptance.csv
runtime_notes/alt-track_t0.15_verbose1000_commtimeout.log
```

## Re-run

Run from the repo root:

```bash
bash outputs/acceptance_exp012_H20_20260824/scripts/run_metric1_latency.sh
bash outputs/acceptance_exp012_H20_20260824/scripts/run_metric2_standard_h20.sh
bash outputs/acceptance_exp012_H20_20260824/scripts/run_7_condition_h20_matrix.sh
```

The scripts write fresh outputs under:

```text
outputs/acceptance_exp012_H20_20260824/rerun/
```

