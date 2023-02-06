#!/bin/bash
#SBATCH --job-name=fine_livus     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=99:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/fine_livus.out # output file name # add %j to id the job
#SBATCH --error=logs/fine_livus.err  # error file name # add %j to id the job
# # SBATCH -C v100-32g


cd $WORK/fine
module purge
#conda deactivate

# conda deactivate
module load cuda/10.2
module load python/3.9.12

# conda activate fine


export nnUNet_raw_data_base="/gpfsscratch/rech/arf/unm89rb/nnUNetData/nnUNet_raw"
export nnUNet_preprocessed="/gpfsscratch/rech/arf/unm89rb/nnUNet_preprocessed"
export RESULTS_FOLDER="/gpfsscratch/rech/arf/unm89rb/nnUNet_trained_models"



# convert dataset
# srun python nnUNet/nnunet/dataset_conversion/Task017_BeyondCranialVaultAbdominalOrganSegmentation.py
# srun python nnUNet/nnunet/dataset_conversion/Task130_Livus.py

# planning and pre-processing
# srun python nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 017 --verify_dataset_integrity
# srun python nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 130 --verify_dataset_integrity


# training
# srun python fine_package/fine/run/run_all_unet.py #unet_livus & unet_livus_bis
# srun python fine_package/fine/run/run_all_cotr.py #cotr_livus
# srun python fine_package/fine/run/run_all_nnformer.py #nnformer_livus
srun python fine_package/fine/run/run_all_fine.py #fine_livus
