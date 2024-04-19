#!/bin/bash 
set -e
mmselfsup_config=( 
configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20_l1loss.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds5_pt_90_ft_10.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds2_pt_80_ft_20.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds10_pt_80_ft_20.py
configs/selfsup/crop_size/mae_ai4arctic_cs256_ds5_pt_80_ft_20.py
configs/selfsup/crop_size/mae_ai4arctic_cs1024_ds5_pt_80_ft_20.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds5_pt_95_ft_5.py
configs/selfsup/ai4arctic/mae_ai4arctic_ds5_pt_50_ft_50.py
configs/selfsup/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr50.py
configs/selfsup/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr90.py
configs/selfsup/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr25.py
)

mmseg_config=( 
configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
configs/ai4arctic/upernet_vit_finetune_neck_replaced.py
configs/ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py
configs/ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20_l1loss.py
configs/ai4arctic/mae_ai4arctic_ds5_pt_90_ft_10.py
configs/ai4arctic/mae_ai4arctic_ds2_pt_80_ft_20.py
configs/ai4arctic/mae_ai4arctic_ds10_pt_80_ft_20.py
configs/crop_size/mae_ai4arctic_cs256_pt_80_ft_20.py
configs/crop_size/mae_ai4arctic_cs1024_pt_80_ft_20.py
configs/ai4arctic/mae_ai4arctic_ds5_pt_95_ft_5.py
configs/ai4arctic/mae_ai4arctic_ds5_pt_50_ft_50.py
configs/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr50.py
configs/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr90.py
configs/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr25.py
configs/ai4arctic/mae_ai4arctic_ds10_pt_80_ft_20.py
)

for i in "${!mmseg_config[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch pretrain_finetune.sh ${mmselfsup_config[i]} ${mmseg_config[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 5
done
