# Dataset ascad_100: Analysis of 100 experiments

## Original (i.e. without early stopping)

![Distribution of min traces needed for average rank to be zero](../plots/ascad_100/violin_no_es.svg)

|eff_cnn<br><span style='color:red'> **1.00 % FAILED** </span>|simplified_eff_cnn<br><span style='color:red'> **6.00 % FAILED** </span>|
|---|---|
|![Average Rank](../plots/ascad_100/eff_cnn/no_es/average_rank.svg)|![Average Rank](../plots/ascad_100/simplified_eff_cnn/no_es/average_rank.svg)|
|![Train Loss](../plots/ascad_100/eff_cnn/no_es/train_loss.svg)|![Train Loss](../plots/ascad_100/simplified_eff_cnn/no_es/train_loss.svg)|
|![Validation Loss](../plots/ascad_100/eff_cnn/no_es/val_loss.svg)|![Validation Loss](../plots/ascad_100/simplified_eff_cnn/no_es/val_loss.svg)|
## Modified original to work with early stopping

![Distribution of min traces needed for average rank to be zero](../plots/ascad_100/violin_es.svg)

|eff_cnn<br><span style='color:red'> **14.00 % FAILED** </span>|simplified_eff_cnn<br><span style='color:red'> **2.00 % FAILED** </span>|
|---|---|
|![Average Rank](../plots/ascad_100/eff_cnn/es/average_rank.svg)|![Average Rank](../plots/ascad_100/simplified_eff_cnn/es/average_rank.svg)|
|![Train Loss](../plots/ascad_100/eff_cnn/es/train_loss.svg)|![Train Loss](../plots/ascad_100/simplified_eff_cnn/es/train_loss.svg)|
|![Validation Loss](../plots/ascad_100/eff_cnn/es/val_loss.svg)|![Validation Loss](../plots/ascad_100/simplified_eff_cnn/es/val_loss.svg)|