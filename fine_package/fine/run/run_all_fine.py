import run_training as run

gpu = '0'

# BCV
# Training
# run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='017', fold=0, outpath='FINE', val=False, npz=True)
# Evaluation
# run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='017', fold=0, outpath='FINE', val=True,  npz=True)


# LIVUS
# run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='130', fold=0, outpath='FINE', val=False, npz=True)
# run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='130', fold=0, outpath='FINE', val=True,  npz=True)


# WORD
run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='140', fold=0, outpath='FINE', val=False, npz=True)
# run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='140', fold=0, outpath='FINE', val=False, npz=True, c=True)
run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='140', fold=0, outpath='FINE', val=True,  npz=True)
