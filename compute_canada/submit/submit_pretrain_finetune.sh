#!/bin/bash 
set -e
mmpretrain_config=( 
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
configs/selfsup/ai4arctic/mae_vit-base-p16.py
)

mmselfsup_config=( 
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
# configs/ai4arctic/upernet_vit_finetune_neck_replaced.py
configs/ai4arctic/mae_ai4arctic_finetune.py
)


for i in "${!mmpretrain_config[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch pretrain_finetune.sh ${mmpretrain_config[i]} ${mmselfsup_config[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 5
done



# for i in "${!mmpretrain_config[@]}"; do
#     echo "train.sh ${mmpretrain_config[i]} ${mmselfsup_config[i]}"
#     echo "task successfully submitted"
#     sleep 5
# done