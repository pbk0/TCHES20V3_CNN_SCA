# Dataset dpav4: Analysis of 100 experiments

## Original (i.e. without early stopping)

![Distribution of min traces needed for average rank to be zero](plots/dpav4/violin_no_es.svg)

|eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|simplified_eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|
|---|---|
|![Average Rank](plots/dpav4/eff_cnn/no_es/average_rank.svg)|![Average Rank](plots/dpav4/simplified_eff_cnn/no_es/average_rank.svg)|
|![Rank Variance](plots/dpav4/eff_cnn/no_es/rank_variance.svg)|![Rank Variance](plots/dpav4/simplified_eff_cnn/no_es/rank_variance.svg)|
|![Train Loss](plots/dpav4/eff_cnn/no_es/train_loss.svg)|![Train Loss](plots/dpav4/simplified_eff_cnn/no_es/train_loss.svg)|
|![Validation Loss](plots/dpav4/eff_cnn/no_es/val_loss.svg)|![Validation Loss](plots/dpav4/simplified_eff_cnn/no_es/val_loss.svg)|
## Modified original to work with early stopping

![Distribution of min traces needed for average rank to be zero](plots/dpav4/violin_es.svg)

|eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|simplified_eff_cnn<br><span style='color:green'> **ALL PASSES** </span>|
|---|---|
|![Average Rank](plots/dpav4/eff_cnn/es/average_rank.svg)|![Average Rank](plots/dpav4/simplified_eff_cnn/es/average_rank.svg)|
|![Rank Variance](plots/dpav4/eff_cnn/es/rank_variance.svg)|![Rank Variance](plots/dpav4/simplified_eff_cnn/es/rank_variance.svg)|
|![Train Loss](plots/dpav4/eff_cnn/es/train_loss.svg)|![Train Loss](plots/dpav4/simplified_eff_cnn/es/train_loss.svg)|
|![Validation Loss](plots/dpav4/eff_cnn/es/val_loss.svg)|![Validation Loss](plots/dpav4/simplified_eff_cnn/es/val_loss.svg)|