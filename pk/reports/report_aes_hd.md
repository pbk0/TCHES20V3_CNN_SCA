# Dataset aes_hd: Analysis of 100 experiments

## Original (i.e. without early stopping)

![Distribution of min traces needed for average rank to be zero](../plots/aes_hd/violin_no_es.svg)

|eff_cnn<br><span style='color:green'> **ALL PASSED** </span>|simplified_eff_cnn<br><span style='color:green'> **ALL PASSED** </span>|
|---|---|
|![Average Rank](../plots/aes_hd/eff_cnn/no_es/average_rank.svg)|![Average Rank](../plots/aes_hd/simplified_eff_cnn/no_es/average_rank.svg)|
|![Train Loss](../plots/aes_hd/eff_cnn/no_es/train_loss.svg)|![Train Loss](../plots/aes_hd/simplified_eff_cnn/no_es/train_loss.svg)|
|![Validation Loss](../plots/aes_hd/eff_cnn/no_es/val_loss.svg)|![Validation Loss](../plots/aes_hd/simplified_eff_cnn/no_es/val_loss.svg)|
## Modified original to work with early stopping

![Distribution of min traces needed for average rank to be zero](../plots/aes_hd/violin_es.svg)

|eff_cnn<br><span style='color:green'> **ALL PASSED** </span>|simplified_eff_cnn<br><span style='color:green'> **ALL PASSED** </span>|
|---|---|
|![Average Rank](../plots/aes_hd/eff_cnn/es/average_rank.svg)|![Average Rank](../plots/aes_hd/simplified_eff_cnn/es/average_rank.svg)|
|![Train Loss](../plots/aes_hd/eff_cnn/es/train_loss.svg)|![Train Loss](../plots/aes_hd/simplified_eff_cnn/es/train_loss.svg)|
|![Validation Loss](../plots/aes_hd/eff_cnn/es/val_loss.svg)|![Validation Loss](../plots/aes_hd/simplified_eff_cnn/es/val_loss.svg)|