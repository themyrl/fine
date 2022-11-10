import run_training as run

g = '0'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGood', task='017', fold=1, outpath='NNFORMERg', val=False, npz=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGood', task='017', fold=1, outpath='NNFORMERg', val=True,  npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGood', task='017', fold=2, outpath='NNFORMERg', val=False, npz=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_nnFormerGood', task='017', fold=2, outpath='NNFORMERg', val=True,  npz=True)

# run.main(gpu=g, network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='017', fold=2, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu=g, network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='017', fold=2, outpath='MHSA_v2', val=True, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=4, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=4, outpath='MHSA_v2', val=True, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=0, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=0, outpath='MHSA_v2', val=True, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=3, outpath='MHSA_v2', val=False, npz=True)
# run.main(gpu='1', network='2d', network_trainer='nnUNetTrainerV2_utrans_mhsa', task='062', fold=3, outpath='MHSA_v2', val=True, npz=True)
