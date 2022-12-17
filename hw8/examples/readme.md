The following example outputs that should be close-ish. Note they may not be exact because of randomness.
- raw_weight.out
- raw_returns.out
- tile_weight.out
- tile_returns.out

parameters used:
- mc raw raw_weight.out raw_returns.out 20 200 0.05 0.99 0.01
- mc tile tile_weight.out tile_returns.out 20 200 0.05 0.99 0.00005

The following example output should match with your output up till the last 4 digits, due to rounding error. There is no randomness in generating the output.
- fixed_weight.out
- fixed_returns.out

parameters used:
- mc tile fixed_weight.out fixed_returns.out 25 200 0.0 0.99 0.005
