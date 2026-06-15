# Loss Time Weight Functions

All functions below are timestep-only weights. They are applied once as the
outer loss weight:

```text
loss(t) = time_weight(t) * sum_i(coef_i * raw_loss_i(t))
```

The examples use the flow schedule `alpha=1-t`, `sigma=t`, and show weights
after mean normalization. The reference points are:

```text
t: 0.001  0.10  0.20  0.25  0.30  0.40  0.50  0.60  0.70  0.75  0.80  0.90  0.999
```

## none

```yaml
loss:
  outer_weight: none
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
none: 1  1  1  1  1  1  1  1  1  1  1  1  1
```

## triangular

Middle-time emphasis. Small near both endpoints, large near `t=0.5`.

```yaml
loss:
  outer_weight: triangular
  outer_weight_power: 1.0
  outer_weight_floor: 0.0
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
triangular: 0.004  0.400  0.800  1.000  1.200  1.600  2.000  1.600  1.200  1.000  0.800  0.400  0.004
```

## skew_triangular

Middle-time emphasis with asymmetric side preference. Positive `skew` makes
the `t -> 1` side heavier than the `t -> 0` side.

```yaml
loss:
  outer_weight: skew_triangular
  outer_weight_power: 1.0
  outer_weight_floor: 0.0
  outer_weight_skew: 0.5
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
skew_triangular(skew=0.5): 0.002  0.240  0.560  0.750  0.960  1.440  2.000  1.760  1.440  1.250  1.040  0.560  0.006
```

More skew examples with the same `power=1.0` and `floor=0.0`:

```yaml
loss:
  outer_weight: skew_triangular
  outer_weight_power: 1.0
  outer_weight_floor: 0.0
  outer_weight_skew: 0.2  # or 0.5 / 0.8
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
skew_triangular(skew=0.2): 0.003  0.336  0.704  0.900  1.100  1.536  2.000  1.664  1.296  1.100  0.896  0.464  0.005
skew_triangular(skew=0.5): 0.002  0.240  0.560  0.750  0.960  1.440  2.000  1.760  1.440  1.250  1.040  0.560  0.006
skew_triangular(skew=0.8): 0.001  0.144  0.416  0.600  0.816  1.344  2.000  1.856  1.584  1.400  1.184  0.656  0.007
```

## p2

SNR-based P2-style weighting. With the settings below, high-noise times
receive larger weight.

```yaml
loss:
  outer_weight: p2
  outer_weight_p2_k: 1.0
  outer_weight_p2_gamma: 1.0
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
p2(k=1.0,gamma=1.0): 2e-6  0.024  0.118  0.200  0.310  0.615  1.000  1.385  1.690  1.800  1.882  1.976  2.000
```

## min_snr

Min-SNR weighting. Low-noise/high-SNR times are suppressed, and middle/high
noise times approach a plateau.

```yaml
loss:
  outer_weight: min_snr
  min_snr_gamma: 5.0
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
min_snr(gamma=5.0): 6e-6  0.080  0.403  0.716  1.183  1.288  1.288  1.288  1.288  1.288  1.288  1.288  1.288
```

## edm

EDM-style sigma weighting. This is usually meant to be used with EDM/sigma
parameterization. Under the simple flow mapping `sigma_ratio=t/(1-t)`, it is
extremely concentrated near `t -> 0`.

```yaml
loss:
  outer_weight: edm
  outer_weight_sigma_data: 0.5
  terms:
    - {target: v, formula: mse, coef: 1.0}
```

```text
edm(sigma_data=0.5): 101.0  8.6e-3  2.0e-3  1.3e-3  9.6e-4  6.3e-4  5.1e-4  4.5e-4  4.3e-4  4.2e-4  4.1e-4  4.1e-4  4.1e-4
```

## Notes

- `outer_weight` is a global per-example timestep weight.
- `coef` controls relative weighting between loss targets.
- Per-term `weight` is still supported for compatibility, but new configs
  should prefer a single `outer_weight`.
- During training, the logger prints a compact text table of the active weight
  shape at startup.
