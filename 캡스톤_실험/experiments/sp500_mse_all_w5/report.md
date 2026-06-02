# Capstone Experiment Report

- created_at: 2026-06-02T18:57:56
- model_type: mse
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
- avg_rank_ic: 0.5404

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      7     0.203526           2.050882 0.172745 -0.072223 0.663201     34.428571   34.428571                    0                    0
gap_top_20pct      7     0.193015           1.643870 0.149000 -0.053999 0.506838     69.285714   69.285714                    0                    0
gap_top_30pct      7     0.170574           1.403120 0.133435 -0.044302 0.501091    103.857143  103.857143                    0                    0
 gap_top_5pct      7     0.166114           1.589208 0.145576 -0.054795 0.662930     16.857143   16.857143                    0                    0
```

## Monthly Portfolio Returns

- rows: 336
- months: 2019-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok     84       0.001853          0.022463     0.864641     1.030539  0.165944  0.114030  0.163770  0.135426             0.014945                    0.012754   0.894485
gap_top_20pct     ok     84      -0.000008         -0.000095    -0.004341     1.035238  0.319749  0.206052  0.283707  0.089297             0.013375                    0.011185   0.930754
gap_top_30pct     ok     84      -0.000986         -0.011773    -0.650335     1.004771  0.291100  0.255567  0.255805  0.072950             0.012121                    0.009930   0.948578
 gap_top_5pct     ok     84      -0.000457         -0.005470    -0.187812     1.011027  0.071530 -0.031413  0.305526  0.107170             0.012893                    0.010702   0.857880
```