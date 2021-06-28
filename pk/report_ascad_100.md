# Dataset ascad_100: Analysis of 100 experiments

## Original (i.e. without early stopping)

![Distribution of min traces needed for average rank to be zero](plots/ascad_100/violin_no_es.svg)

|ascad_mlp<br><span style='color:red'> **100.00 % FAILED** </span>|ascad_mlp_fn<br><span style='color:red'> **100.00 % FAILED** </span>|eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|simplified_eff_cnn<br><span style='color:red'> **10.00 % FAILED** </span>|
|---|---|---|---|
|![Average Rank](plots/ascad_100/ascad_mlp/no_es/average_rank.svg)|![Average Rank](plots/ascad_100/ascad_mlp_fn/no_es/average_rank.svg)|![Average Rank](plots/ascad_100/eff_cnn/no_es/average_rank.svg)|![Average Rank](plots/ascad_100/simplified_eff_cnn/no_es/average_rank.svg)|
|![Rank Variance](plots/ascad_100/ascad_mlp/no_es/rank_variance.svg)|![Rank Variance](plots/ascad_100/ascad_mlp_fn/no_es/rank_variance.svg)|![Rank Variance](plots/ascad_100/eff_cnn/no_es/rank_variance.svg)|![Rank Variance](plots/ascad_100/simplified_eff_cnn/no_es/rank_variance.svg)|
|![Train Loss](plots/ascad_100/ascad_mlp/no_es/train_loss.svg)|![Train Loss](plots/ascad_100/ascad_mlp_fn/no_es/train_loss.svg)|![Train Loss](plots/ascad_100/eff_cnn/no_es/train_loss.svg)|![Train Loss](plots/ascad_100/simplified_eff_cnn/no_es/train_loss.svg)|
|![Validation Loss](plots/ascad_100/ascad_mlp/no_es/val_loss.svg)|![Validation Loss](plots/ascad_100/ascad_mlp_fn/no_es/val_loss.svg)|![Validation Loss](plots/ascad_100/eff_cnn/no_es/val_loss.svg)|![Validation Loss](plots/ascad_100/simplified_eff_cnn/no_es/val_loss.svg)|
## Modified original to work with early stopping

![Distribution of min traces needed for average rank to be zero](plots/ascad_100/violin_es.svg)

|ascad_mlp<br><span style='color:red'> **100.00 % FAILED** </span>|ascad_mlp_fn<br><span style='color:red'> **100.00 % FAILED** </span>|eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|simplified_eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|
|---|---|---|---|
|![Average Rank](plots/ascad_100/ascad_mlp/es/average_rank.svg)|![Average Rank](plots/ascad_100/ascad_mlp_fn/es/average_rank.svg)|![Average Rank](plots/ascad_100/eff_cnn/es/average_rank.svg)|![Average Rank](plots/ascad_100/simplified_eff_cnn/es/average_rank.svg)|
|![Rank Variance](plots/ascad_100/ascad_mlp/es/rank_variance.svg)|![Rank Variance](plots/ascad_100/ascad_mlp_fn/es/rank_variance.svg)|![Rank Variance](plots/ascad_100/eff_cnn/es/rank_variance.svg)|![Rank Variance](plots/ascad_100/simplified_eff_cnn/es/rank_variance.svg)|
|![Train Loss](plots/ascad_100/ascad_mlp/es/train_loss.svg)|![Train Loss](plots/ascad_100/ascad_mlp_fn/es/train_loss.svg)|![Train Loss](plots/ascad_100/eff_cnn/es/train_loss.svg)|![Train Loss](plots/ascad_100/simplified_eff_cnn/es/train_loss.svg)|
|![Validation Loss](plots/ascad_100/ascad_mlp/es/val_loss.svg)|![Validation Loss](plots/ascad_100/ascad_mlp_fn/es/val_loss.svg)|![Validation Loss](plots/ascad_100/eff_cnn/es/val_loss.svg)|![Validation Loss](plots/ascad_100/simplified_eff_cnn/es/val_loss.svg)|