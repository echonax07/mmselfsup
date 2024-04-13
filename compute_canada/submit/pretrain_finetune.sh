#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=p100:2 # request a GPU
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=6 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=50G
#SBATCH --time=2:59:00
#SBATCH --output=../output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
set -e

module purge
module load python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"

source ~/env_mmselfsup/bin/activate

echo "Activating virtual environment done"

cd $HOME/projects/def-dclausi/AI4arctic/$USER/mmselfsup

echo "starting training..."

export WANDB_MODE=offline

echo "Config file: $1"
# srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm --resume
# srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm
srun --ntasks=2 --gpus-per-node=p100:2  --kill-on-bad-exit=1 --cpus-per-task=6 python tools/train.py $1 --launcher slurm

# Extract the base name without extension
base_name=$(basename "$1" .py)
CHECKPOINT=$(cat work_dirs/selfsup/$base_name/last_checkpoint)
echo "mmpretrain Checkpoint $CHECKPOINT"

# train file 1
python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_train_v3/S1B_EW_GRDM_1SDH_20211119T080313_20211119T080413_029654_0389FB_9317_icechart_dmi_202111190805_CentralEast_RIC.nc"  --out-file "work_dirs/selfsup/$base_name/20211117T212054"


# train file 2
python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_train_v3/S1B_EW_GRDM_1SDH_20191018T130839_20191018T130933_018530_022EA4_4267_icechart_cis_SGRDIMID_20191018T1302Z_pl_a.nc"  --out-file "work_dirs/selfsup/$base_name/20211117T212054"


# test file 1
python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_test_v3/S1B_EW_GRDM_1SDH_20210328T202742_20210328T202842_026220_032117_57E3_icechart_dmi_202103282025_CapeFarewell_RIC.nc"  --out-file "work_dirs/selfsup/$base_name/20210328T202742"


# test file 2 
python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_test_v3/S1A_EW_GRDM_1SDH_20210430T205436_20210430T205537_037685_047252_CBB0_icechart_dmi_202104302055_SouthWest_RIC.nc"  --out-file "work_dirs/selfsup/$base_name/20210430T205436"

deactivate

cd $HOME/projects/def-dclausi/AI4arctic/$USER/mmsegmentation

source ~/env_mmsegmentation/bin/activate

echo "Config file: $2"

srun --ntasks=2 --gpus-per-node=p100:2  --kill-on-bad-exit=1 --cpus-per-task=6 python tools/train.py $2 --launcher slurm --cfg-options  model.pretrained=${CHECKPOINT}

# Extract the base name without extension
base_name_mmseg=$(basename "$2" .py)
CHECKPOINT_mmseg=$(cat work_dirs/$base_name_mmseg/last_checkpoint)
echo "mmseg checkpoint $CHECKPOINT_mmseg"

python tools/test.py $2 $CHECKPOINT_mmseg --out work_dirs/$base_name_mmseg/ --show-dir work_dirs/$base_name_mmseg/