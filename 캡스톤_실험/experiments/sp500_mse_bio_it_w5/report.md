# Capstone Experiment Report

- created_at: 2026-06-02T18:58:00
- model_type: mse
- sector_scope: bio_it
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
- avg_rank_ic: 0.4743

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      7     0.433223           8.604190 0.381502 -0.054506 1.053555      5.571429    5.571429                    0                    0
gap_top_20pct      7     0.367777           5.980181 0.319935 -0.061420 0.942222     11.571429   11.571429                    0                    0
gap_top_30pct      7     0.297757           4.060240 0.260654 -0.035613 0.879766     17.571429   17.571429                    0                    0
 gap_top_5pct      7     0.423980           9.220376 0.393829 -0.015323 1.373809      2.571429    2.571429                    0                    0
```

## Monthly Portfolio Returns

- rows: 336
- months: 2019-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok     84       0.015454          0.202047     2.707835     1.249787 -0.192443 -0.100462 -0.008240 -0.094671             0.030340                    0.028150   0.612651
gap_top_20pct     ok     84       0.011440          0.146261     2.574415     1.244784 -0.044252 -0.020134 -0.037690 -0.053087             0.026022                    0.023831   0.724537
gap_top_30pct     ok     84       0.008156          0.102381     2.127082     1.154785  0.001993 -0.052364 -0.058743  0.016160             0.021651                    0.019460   0.752748
 gap_top_5pct     ok     84       0.016477          0.216663     2.614708     1.252882 -0.200705 -0.017627  0.061921 -0.469698             0.031550                    0.029360   0.589392
```