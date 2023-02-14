import run_training as run

g = '0'
# BCV
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGood', task='017', fold=0, outpath='NNFORMERg', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGood', task='017', fold=0, outpath='NNFORMERg', val=True,  npz=True)


# LIVUS
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='130', fold=0, outpath='NNFORMER', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='130', fold=0, outpath='NNFORMER', val=True,  npz=True)

# WORD
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='140', fold=0, outpath='NNFORMER', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='140', fold=0, outpath='NNFORMER', val=True,  npz=True)
