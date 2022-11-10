import run_training as run

gpu = '2'

# Training
run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_fine', task='017', fold=0, outpath='FINE', val=False, npz=True)

# Evaluation
run.main(gpu=gpu, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=0, outpath='NNFORMEREXTGT1V6', val=True,  npz=True)
