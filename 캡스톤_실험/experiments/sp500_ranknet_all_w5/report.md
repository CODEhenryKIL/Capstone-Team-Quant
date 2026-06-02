# Capstone Experiment Report

- created_at: 2026-06-02T18:58:05
- model_type: ranknet
- sector_scope: all
- train_window_years: 5
- ridge_alpha: 1.0
- ranknet_epochs: 150
- ranknet_max_pairs_per_year: 5000
- y_target_mode: sector
- bio_it_definition: Sector in ('Information Technology',) or Industry in ('Biotechnology',)
- membership_rule: selected from FiscalYear S&P 500 members; held positions are not removed just because the ticker exits S&P 500 later
- evaluation_policy: only completed holding windows are evaluated; incomplete final windows are excluded
- delisting_policy: S&P 500 exits keep using available market prices; missing sell price inside the available price-history horizon is treated as -100%
- evaluation_tracks: raw annual backtest plus optional Fama-French 5-factor alpha
- ff5_factor_file: data/raw/fama_french_5_factors_monthly.csv

## Training

- prediction_years: 2018-2025
- avg_rank_ic: 0.5359

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      7     0.178176           1.624651 0.147803 -0.072223 0.582915     34.428571   34.428571                    0                    0
gap_top_20pct      7     0.196038           1.718308 0.153567 -0.046498 0.523763     69.285714   69.285714                    0                    0
gap_top_30pct      7     0.173626           1.452276 0.136718 -0.045816 0.510020    103.857143  103.857143                    0                    0
 gap_top_5pct      7     0.166281           1.592467 0.145782 -0.054795 0.664084     16.857143   16.857143                    0                    0
```

## Monthly Portfolio Returns

- rows: 336
- months: 2019-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok     84      -0.000181         -0.002170    -0.092298     1.035932  0.197916  0.105106  0.236828  0.133120             0.013151                    0.010960   0.912568
gap_top_20pct     ok     84       0.000480          0.005775     0.263639     1.016468  0.314174  0.198416  0.272370  0.103535             0.013641                    0.011451   0.928152
gap_top_30pct     ok     84      -0.000696         -0.008323    -0.450019     0.998669  0.289737  0.234457  0.258956  0.078396             0.012332                    0.010142   0.945570
 gap_top_5pct     ok     84      -0.000390         -0.004675    -0.160555     1.005829  0.075872 -0.030827  0.306743  0.102050             0.012900                    0.010710   0.857095
```