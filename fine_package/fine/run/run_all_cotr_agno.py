import run_training as run

# g = '5'
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR_agno', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR_agno', val=True, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_agno', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='COTR_agno', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='COTR_agno', val=False, npz=True, na=True,c=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='COTR_agno', val=True, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='COTR_agno', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='COTR_agno', val=True, npz=True, na=True)


g = '2'
# g = '5'
# g = '0'

# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=2, outpath='nnffCOTR_agno', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=3, outpath='nnffCOTR_agno', val=True, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='nnffCOTR_agno', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=1, outpath='nnffCOTR_agno', val=True, npz=True, na=True)


# g='1'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_CoTR_agno', task='017', fold=0, outpath='COTR_agno', val=False, npz=True, na=True,dbg=True)
