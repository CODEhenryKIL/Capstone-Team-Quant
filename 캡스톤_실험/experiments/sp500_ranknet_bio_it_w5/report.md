# Capstone Experiment Report

- created_at: 2026-06-02T18:58:09
- model_type: ranknet
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
- avg_rank_ic: 0.4585

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      7     0.518105          12.160136 0.445088 -0.054506 0.974688      5.571429    5.571429                    0                    0
gap_top_20pct      7     0.336260           5.000882 0.291735 -0.081048 0.897894     11.571429   11.571429                    0                    0
gap_top_30pct      7     0.292012           3.924327 0.255760 -0.062415 0.875079     17.571429   17.571429                    0                    0
 gap_top_5pct      7     0.414825           8.822395 0.385943 -0.015323 1.374432      2.571429    2.571429                    0                    0
```

## Monthly Portfolio Returns

- rows: 336
- months: 2019-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok     84       0.020234          0.271744     3.112853     1.239497 -0.150260 -0.064885 -0.142780 -0.106143             0.034484                    0.032294   0.546639
gap_top_20pct     ok     84       0.009864          0.125009     2.222982     1.229596 -0.076429 -0.032704 -0.082305 -0.103435             0.024139                    0.021949   0.720774
gap_top_30pct     ok     84       0.008131          0.102055     2.327068     1.146146  0.068664 -0.101892 -0.059240 -0.042809             0.021271                    0.019080   0.789889
 gap_top_5pct     ok     84       0.015555          0.203494     2.548408     1.307925 -0.198926 -0.186718  0.148752 -0.437705             0.031232                    0.029042   0.632116
```