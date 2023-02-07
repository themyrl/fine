import run_training as run

g = '0'
# BCV
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR', val=True, npz=True, na=True)

# LIVUS
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='130', fold=0, outpath='COTR', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='130', fold=0, outpath='COTR', val=True, npz=True, na=True)

# WORD
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='140', fold=0, outpath='COTR', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='140', fold=0, outpath='COTR', val=True, npz=True, na=True)




