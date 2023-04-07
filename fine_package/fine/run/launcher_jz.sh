#!/bin/bash
#SBATCH --job-name=cofi_word     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=20   #10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/cofi_word.out # output file name # add %j to id the job
#SBATCH --error=logs/cofi_word.err  # error file name # add %j to id the job
#SBATCH -C v100-32g
 

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

### WORD
# srun python fine_package/fine/run/run.py nnUNetTrainerV2 140 notta_NNUNET 1 0 0 #unet_word
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_dft 140 notta_NNUNET_dft 1 0 0 #unet_word_dft
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_nnFormer 140 notta_NNFORMER 0 0 0 #nnfo_word
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fine 140 notta_FINE 0 0 0 #fine_word
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet 140 notta_FINENNUNET 1 0 0 #finun_word
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_CoTR_agno 140 notta_COTR 1 0 0 #cotr_word

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet_v2 140 notta_FINENNUNETV2 1 0 0 #finunv2_word # we added fine module at each layer of the encoder
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet_v2 140 notta_FINENNUNETV2 1 0 0 1 #finunv2_word_c # we added fine module at each layer of the encoder

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet_v3 140 notta_FINENNUNETV3 1 0 0 #finunv3_word # we added fine module at each layer of the encoder and skip connection after transformer module

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet_v2 140 notta_FINENNUNETV2_1 1 0 0 #finunv21_word # 3 block in each layer

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_glam 140 notta_GLAM 0 0 0 #glam_word
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glam -task 140 -outpath GLAMV2 -continu #glamv2_word_c

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_finev3 140 notta_FINEV3 0 0 0 #finv3_word
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3 -task 140 -outpath notta_FINEV3 -continu #funv3v2_livus_c # fine+nnunet with fine v3 at all encoder stage

# srun python fine_package/fine/run/run.py  -network nnUNetTrainerV2_finev3 -task 140 -outpath notta_FINEV32 #finv32_word
# srun python fine_package/fine/run/run.py  -network nnUNetTrainerV2_finev3 -task 140 -outpath notta_FINEV32 -continu #finv32_word_c



# srun python fine_package/fine/run/run.py nnUNetTrainerV2_finev3UNet_v2 140 notta_FINEV3NNUNETV2 1 0 0 #funv3v2_word # fine+nnunet with fine v3 at almost all encoder stage
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3UNet_v2 -task 140 -outpath notta_FINEV32NNUNETV2 -na #fi32u2_word # fine v32+nnunet v2 with fine v3 at almost all encoder stage


# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glamUNet_v2 -task 140 -outpath notta_GLAMV2NNUNET -na #glv2un_word # glamv2+nnunet with glam at almost all encoder stage


srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_CoTR_FINE -task 140 -outpath notta_COTRFINE -na # cofi_word



### LIVUS
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_finedbg 130 DBGFINE 0 0 1 #debug
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_finedbg 130 DBGFINE 0 0 1 1 #cdebug
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_finedbg 130 DBGFINE 0 1 1 1 #edebug

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fine_us 130 FINEUS 0 0 1 #finus_livus


# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glam -task 130 -outpath GLAM -tta -clip #glam_livus
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glam -task 130 -outpath GLAMV2 -tta -clip #glamv2_livus
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glamUNet_v2 -task 130 -outpath GLAMNNUNETV2 -na -tta -clip #glunv2_livus # glam+nnunet with glam at almost all encoder stage
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glamUNet_v2 -task 130 -outpath GLAMV2NNUNET -na -tta #glv2un_livus # glamv2+nnunet with glam at almost all encoder stage


# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3 -task 130 -outpath FINEV3 -tta -clip #finv3_livus
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3 -task 130 -outpath FINEV32 -tta #finv32_livus
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3 -task 130 -outpath FINEV32 -tta -continu #finv32_livus_c


# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3UNet_v2 -task 130 -outpath FINEV3NNUNETV2 -na -tta -clip #funv3v2_livus # fine+nnunet with fine v3 at all encoder stage
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3UNet_v2 -task 130 -outpath FINEV32NNUNETV2 -na -tta #fi32u2_livus # fine+nnunet with fine v3 at all encoder stage




## Only eval
# srun python fine_package/fine/run/run_all_fineunet.py nnUNetTrainerV2_fineUNet 140 FINENNUNET 1 1 #finun_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2 140 NNUNET_128_128_64 1 1 1 #unet_word_128_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_nnFormer 140 NNFORMER 0 1 1 #nnfo_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fine 140 FINE 0 1 1 #fine_word_eval

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_dft 140 notta_NNUNET_dft 1 1 0 #unet_word_dft_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2 140 notta_NNUNET 1 1 0 #unet_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_nnFormer 140 notta_NNFORMER 0 1 0 #nnfo_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fine 140 notta_FINE 0 1 0 #fine_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet 140 notta_FINENNUNET 1 1 0 #finun_word_eval
# srun python fine_package/fine/run/run.py nnUNetTrainerV2_CoTR_agno 140 notta_COTR 1 1 0 #cotr_word_eval

# srun python fine_package/fine/run/run.py nnUNetTrainerV2_fineUNet_v2 140 notta_FINENNUNETV2 1 1 0 1 #finunv2_word_eval # we added fine module at each layer of the encoder

# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3UNet_v2 -task 140 -outpath notta_FINEV3NNUNETV2 -na -only_val #funv3v2_livus_eval # fine+nnunet with fine v3 at all encoder stage
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glamUNet_v2 -task 140 -outpath notta_GLAMNNUNETV2 -na -only_val #glunv2_word_eval # glam+nnunet with glam at almost all encoder stage
# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3 -task 140 -outpath notta_FINEV3 -only_val #funv3v2_word_eval # fine+nnunet with fine v3 at all encoder stage


# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_finev3UNet_v2 -task 140 -outpath notta_FINEV32NNUNETV2 -na -only_val #fi32u2_word_eval # fine v32+nnunet v2 with fine v3 at almost all encoder stage

# srun python fine_package/fine/run/run.py -network nnUNetTrainerV2_glamUNet_v2 -task 140 -outpath notta_GLAMV2NNUNET -na -only_val #glv2un_word_eval # glamv2+nnunet with glam at almost all encoder stage

















