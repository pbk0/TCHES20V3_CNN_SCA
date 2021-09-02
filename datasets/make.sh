wget https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5
unzip AES_HD_dataset.zip
unzip ASCAD_dataset.zip
unzip DPAv4_dataset.zip

cat AES_RD_dataset.z* > AES_RD_dataset_full.zip
unzip AES_RD_dataset_full.zip