import run_training as run

g = '1'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=0, outpath='NNUNETaf', val=True,  npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1gv2', task='017', fold=1, outpath='NNFORMERGT1af', val=True,  npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormer', task='017', fold=1, outpath='NNFORMERaf', val=True,  npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=2, outpath='NNFORMEREXTGT1V6af', val=True,  npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGT1EXTV6', task='017', fold=3, outpath='NNFORMEREXTGT1V6af', val=True,  npz=True)
