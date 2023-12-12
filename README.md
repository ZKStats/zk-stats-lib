# Zk Stats

A library to generate and verify proof for statistical queries to one data provider.

Functions to support

- `mean`
- `geometric_mean`
- `harmonic_mean`
- `median`
- `mode`
- `pstdev`
- `pvariance`
- `stdev`
- `variance`
- `covariance`
- `correlation`
- `linear_regression`

Note:
[to be updated after benchmark test]

- All functions above work, except correlation. It is unnecessary to write correlation's own circuit because it can be derived from covariance and std + its circuit can be too big.
- We implement using witness approach instead of directly calculating the value in circuit. This sometimes allows us to not calculate stuffs like division or exponential which requires larger scale in settings. (If we dont use larger scale in those cases, the accuracy will be very bad)
- For non-linearity function, larger scale leads to larger lookup table, hence bigger circuit size. Can compare between geomean_OG (implemented in traditional way, instead of witness approach) which is the non-linearity function (p bad with larger scale), and mean_OG which is linear function (p fine with larger scale). Hence, we can say that for linearity func like mean, we can use traditional way, while for non-linear func like geomean, we should use witness approach.
- Dummy data to feed in verifier onnx file needs to have same shape as the private dataset, but can be filled with any value (we just randomize it to be uniform 1-10 with 1 decimal).
