# Capstone Experiment Report

- created_at: 2026-06-02T18:58:09
- model_type: ranknet
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
- avg_rank_ic: 0.5100

## Backtest

```text
     Strategy  years  mean_return  cumulative_return     cagr       mdd   sharpe  avg_selected  avg_priced  total_delisted_zero  total_price_missing
gap_top_10pct      9     0.196704           3.163180 0.171723 -0.123543 0.727110     33.111111   33.111111                    0                    0
gap_top_20pct      9     0.178470           2.380250 0.144912 -0.187528 0.543192     66.555556   66.555556                    0                    0
gap_top_30pct      9     0.150545           1.789715 0.120745 -0.205302 0.499770     99.888889   99.888889                    1                    0
 gap_top_5pct      9     0.171338           2.664278 0.155222 -0.069495 0.788620     16.222222   16.222222                    0                    0
```

## Monthly Portfolio Returns

- rows: 432
- months: 2017-04-30 to 2026-03-31
- file: `monthly_portfolio_returns.csv`

## Fama-French 5-Factor Alpha

```text
     Strategy status  n_obs  alpha_monthly  alpha_annualized  alpha_tstat  beta_mkt_rf  beta_smb  beta_hml  beta_rmw  beta_cma  mean_monthly_return  mean_monthly_excess_return  r_squared
gap_top_10pct     ok    108       0.002458          0.029902     1.460452     1.041621  0.159816  0.118864  0.188786  0.109836             0.014723                    0.012737   0.905345
gap_top_20pct     ok    108       0.000671          0.008083     0.465967     1.031268  0.241068  0.212186  0.256764  0.101856             0.012806                    0.010820   0.932657
gap_top_30pct     ok    108      -0.000979         -0.011682    -0.779617     1.018621  0.253588  0.239651  0.254443  0.082067             0.010988                    0.009001   0.947775
 gap_top_5pct     ok    108       0.000873          0.010524     0.430088     1.011206  0.080016 -0.039795  0.297583  0.104113             0.013449                    0.011463   0.855746
```