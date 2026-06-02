# Capstone Experiment Report

- created_at: 2026-06-02T18:58:10
- model_type: ranknet
- sector_scope: bio_it
- train_window_years: 3
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

- prediction_years: 2016-2025
- avg_rank_ic: 0.4381

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      9     0.476853          21.118912 0.410647 -0.054506 0.967596      5.222222    5.222222                    0                    0
gap_top_20pct      9     0.326537           8.897534 0.290072 -0.040583 0.964200     10.888889   10.888889                    0                    0
gap_top_30pct      9     0.283839           6.584218 0.252472 -0.037154 0.918362     16.444444   16.444444                    0                    0
 gap_top_5pct      9     0.401327          16.434340 0.373833 -0.015323 1.365902      2.444444    2.444444                    0                    0
```

## Monthly Portfolio Returns

- rows: 432
- months: 2017-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok    108       0.017626          0.233272     3.220451     1.254212 -0.069480 -0.224277  0.022360 -0.111411             0.032262                    0.030276   0.557587
gap_top_20pct     ok    108       0.010213          0.129676     2.644176     1.186823  0.054919 -0.091775  0.010287 -0.130308             0.023804                    0.021818   0.697373
gap_top_30pct     ok    108       0.008098          0.101620     2.664344     1.127108  0.110744 -0.075547 -0.020408 -0.072940             0.020862                    0.018875   0.771587
 gap_top_5pct     ok    108       0.015166          0.197968     2.602008     1.233728  0.059532 -0.235906  0.347574 -0.525874             0.030461                    0.028475   0.562543
```