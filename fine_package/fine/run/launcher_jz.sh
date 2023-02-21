#!/bin/bash
#SBATCH --job-name=nta_cotr_word_eval     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=99:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/nta_cotr_word_eval.out # output file name # add %j to id the job
#SBATCH --error=logs/nta_cotr_word_eval.err  # error file name # add %j to id the job
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



## convert dataset
# srun python nnUNet/nnunet/dataset_conversion/Task017_BeyondCranialVaultAbdominalOrganSegmentation.py
# srun python nnUNet/nnunet/dataset_conversion/Task130_Livus.py
# srun python nnUNet/nnunet/dataset_conversion/Task140_WORD.py

## planning and pre-processing
# srun python nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 017 --verify_dataset_integrity
# srun python nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 130 --verify_dataset_integrity
# srun python nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 140 --verify_dataset_integrity


## training
# srun python fine_package/fine/run/run_all_unet.py #unet_livus & unet_livus_bis & unet_word+_eval & unet_word_128
# srun python fine_package/fine/run/run_all_cotr.py #cotr_livus & cotr_word+_eval 
# srun python fine_package/fine/run/run_all_nnformer.py #nnformer_livsus & nnfo_word
# srun python fine_package/fine/run/run_all_fine.py #fine_word+_c & fine_livus
# srun python fine_package/fine/run/run_all_fineunet.py #finun_livus & finun_word


# srun python fine_package/fine/run/run_all_fineunet.py nnUNetTrainerV2_fineUNet 130 FINENNUNET 1 0 #finun_livus


## Only eval
# srun python fine_package/fine/run/run_all_fineunet.py nnUNetTrainerV2_fineUNet 140 FINENNUNET 1 1 #finun_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2 140 NNUNET_128_128_64 1 1 1 #unet_word_128_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_nnFormer 140 NNFORMER 0 1 1 #nnfo_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fine 140 FINE 0 1 1 #fine_word_eval

# srun python fine_package/fine/run/run.py nnUNetTrainerV2 140 notta_NNUNET 1 1 0 #nta_unet_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2 140 notta_NNUNET_128_128_64 1 1 0 #nta_unet_word_128_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_nnFormer 140 notta_NNFORMER 0 1 0 #nta_nnfo_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fine 140 notta_FINE 0 1 0 #nta_fine_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet 140 notta_FINENNUNET 0 1 0 #nta_finun_word_eval
srun python fine_package/fine/run/run.py nnUNetTrainerV2_CoTR_agno 140 notta_COTR 1 1 0 #nta_cotr_word_eval



