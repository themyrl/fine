# import UTrans.run.nnunet_run_training as run
import run_training as run

g = '4'
# g = '0'


# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=0, outpath='NNUNETaf', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=0, outpath='NNUNETaf', val=True, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=1, outpath='NNUNETaf', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=1, outpath='NNUNETaf', val=True, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=2, outpath='NNUNETaf', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=2, outpath='NNUNETaf', val=True, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=3, outpath='NNUNETaf', val=False, npz=True, na=True)
# # run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=3, outpath='NNUNETaf', val=True, npz=True, na=True)



run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=2, outpath='nnffNNUNET', val=False, npz=True, na=True)
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=2, outpath='nnffNNUNET', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=3, outpath='nnffNNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=3, outpath='nnffNNUNET', val=True, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=1, outpath='nnffNNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=1, outpath='nnffNNUNET', val=True, npz=True, na=True)

# g='1'
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2', task='017', fold=0, outpath='nnffNNUNET', val=False, npz=True, na=True,dbg=True)
