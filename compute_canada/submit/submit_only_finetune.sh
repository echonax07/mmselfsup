#!/bin/bash 
set -e
array=(
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_90_ft_10_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds2_pt_80_ft_20_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds10_pt_80_ft_20_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_95_ft_5_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_50_ft_50_onlyfinetune.py
# configs/ai4arctic/mae_ai4arctic_ds5_pt_0_ft_100_onlyfinetune.py
configs/different_decoders/mae_ai4arctic_ds5_pt_80_ft_20_onlyfinetune_Mask2FormerHead.py
configs/different_decoders/mae_ai4arctic_ds5_pt_80_ft_20_onlyfinetune_MaskFormerHead.py
configs/different_decoders/mae_ai4arctic_ds5_pt_80_ft_20_onlyfinetune_Segformer.py
configs/different_decoders/mae_ai4arctic_ds5_pt_80_ft_20_onlyfinetune_SegmenterMaskTransformerHead.py
)

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch --exclude=  only_finetune.sh ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 5
done