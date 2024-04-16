#!/bin/bash 
set -e
array=(
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_90_ft_10_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds2_pt_80_ft_20_onlyfinetune.py
configs/ai4arctic/mae_ai4arctic_ds10_pt_80_ft_20_onlyfinetune.py
)

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch only_finetune.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 5
done