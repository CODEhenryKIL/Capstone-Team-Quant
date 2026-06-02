# Capstone Experiment Report

- created_at: 2026-06-02T18:58:00
- model_type: mse
- sector_scope: all
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
- avg_rank_ic: 0.5091

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      9     0.175641           2.576899 0.152128 -0.116006 0.662595     33.111111   33.111111                    0                    0
gap_top_20pct      9     0.164915           2.023574 0.130814 -0.196565 0.498854     66.555556   66.555556                    1                    0
gap_top_30pct      9     0.148016           1.740501 0.118530 -0.211594 0.495014     99.888889   99.888889                    1                    0
 gap_top_5pct      9     0.181060           2.937068 0.164475 -0.069495 0.825202     16.222222   16.222222                    0                    0
```

## Monthly Portfolio Returns

- rows: 432
- months: 2017-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok    108       0.000997          0.012031     0.636331     1.034578  0.155135  0.101448  0.203773  0.130128             0.013253                    0.011266   0.915427
gap_top_20pct     ok    108      -0.000372         -0.004449    -0.251918     1.033600  0.279526  0.212803  0.292133  0.106458             0.011808                    0.009822   0.931420
gap_top_30pct     ok    108      -0.001120         -0.013363    -0.894886     1.014211  0.268111  0.245621  0.274753  0.082416             0.010829                    0.008843   0.948273
 gap_top_5pct     ok    108       0.001557          0.018844     0.756929     1.006844  0.084497 -0.042094  0.310206  0.097759             0.014123                    0.012136   0.851858
```