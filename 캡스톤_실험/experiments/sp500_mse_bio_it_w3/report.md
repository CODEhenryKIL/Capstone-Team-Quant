# Capstone Experiment Report

- created_at: 2026-06-02T18:58:01
- model_type: mse
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
- avg_rank_ic: 0.4225

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      9     0.511783          26.218271 0.443540 -0.054506 1.028944      5.222222    5.222222                    0                    0
gap_top_20pct      9     0.335000           9.129219 0.293393 -0.061420 0.935028     10.888889   10.888889                    0                    0
gap_top_30pct      9     0.276101           6.026644 0.241890 -0.035658 0.858297     16.444444   16.444444                    0                    0
 gap_top_5pct      9     0.392113          15.674594 0.367048 -0.015323 1.415913      2.444444    2.444444                    0                    0
```

## Monthly Portfolio Returns

- rows: 432
- months: 2017-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok    108       0.019583          0.262035     3.454597     1.243879 -0.091237 -0.121456  0.075597 -0.214349             0.034295                    0.032309   0.536962
gap_top_20pct     ok    108       0.010063          0.127673     2.675598     1.222838  0.009709 -0.064609  0.004046 -0.148023             0.024074                    0.022088   0.717754
gap_top_30pct     ok    108       0.007401          0.092516     2.380212     1.116246  0.111121 -0.032613  0.011775 -0.093214             0.020134                    0.018148   0.760384
 gap_top_5pct     ok    108       0.014488          0.188399     2.804457     1.262430 -0.001999 -0.251188  0.212713 -0.501212             0.029789                    0.027803   0.625364
```