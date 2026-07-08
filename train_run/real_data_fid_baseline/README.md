# Real Data FID Baseline

This folder measures the metric floor by treating real CIFAR10 train images as
the generated side (`input1`) and comparing them with torch-fidelity's CIFAR10
reference (`input2`).

Run:

```bash
bash train_run/real_data_fid_baseline/run.sh
```

Defaults:

- `NUM_SAMPLES=50000`
- `INPUT2=cifar10-train`
- `DATASET_ROOT=/root/data/cifar10_data`

Experiments:

- `random_train_50k`: random 50k samples from CIFAR10 train.
- `balanced_train_50k`: class-balanced 50k samples from CIFAR10 train.

For CIFAR10 train, there are exactly 50,000 images with 5,000 per class, so at
`NUM_SAMPLES=50000` both experiments select the full train set. They are kept
separate so smaller `NUM_SAMPLES` runs can expose random-vs-balanced variance.

To compare real train samples against the CIFAR10 test split instead:

```bash
INPUT2=cifar10-val bash train_run/real_data_fid_baseline/run.sh
```

To compare against the EDM CIFAR10 reference statistics used by
`/data/seek/aigc/edm/fid.py`:

```bash
INPUT2=edm-cifar10-32x32 bash train_run/real_data_fid_baseline/run.sh
```

This downloads/caches `cifar10-32x32.npz` and runs `/data/seek/aigc/edm/fid.py
calc` so the feature extractor matches the EDM reference. In this mode the
script reports FID only, because the EDM reference is `(mu, sigma)` statistics
rather than a second image set.
