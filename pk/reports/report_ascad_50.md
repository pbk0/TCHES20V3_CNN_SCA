# Dataset ascad_50: Analysis of 100 experiments

## Original (i.e. without early stopping)

![Distribution of min traces needed for average rank to be zero](../plots/ascad_50/violin_no_es.svg)

|eff_cnn<br><span style='color:green'> **ALL PASSED** </span>|simplified_eff_cnn<br><span style='color:red'> **81.00 % FAILED** </span>|
|---|---|
|![Average Rank](../plots/ascad_50/eff_cnn/no_es/average_rank.svg)|![Average Rank](../plots/ascad_50/simplified_eff_cnn/no_es/average_rank.svg)|
|![Train Loss](../plots/ascad_50/eff_cnn/no_es/train_loss.svg)|![Train Loss](../plots/ascad_50/simplified_eff_cnn/no_es/train_loss.svg)|
|![Validation Loss](../plots/ascad_50/eff_cnn/no_es/val_loss.svg)|![Validation Loss](../plots/ascad_50/simplified_eff_cnn/no_es/val_loss.svg)|
## Modified original to work with early stopping

![Distribution of min traces needed for average rank to be zero](../plots/ascad_50/violin_es.svg)

|eff_cnn<br><span style='color:green'> **ALL PASSED** </span>|simplified_eff_cnn<br><span style='color:red'> **81.00 % FAILED** </span>|
|---|---|
|![Average Rank](../plots/ascad_50/eff_cnn/es/average_rank.svg)|![Average Rank](../plots/ascad_50/simplified_eff_cnn/es/average_rank.svg)|
|![Train Loss](../plots/ascad_50/eff_cnn/es/train_loss.svg)|![Train Loss](../plots/ascad_50/simplified_eff_cnn/es/train_loss.svg)|
|![Validation Loss](../plots/ascad_50/eff_cnn/es/val_loss.svg)|![Validation Loss](../plots/ascad_50/simplified_eff_cnn/es/val_loss.svg)|