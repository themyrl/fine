# import UTrans.run.nnunet_run_training as run
import run_training as run

g = '0'


# BCV
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='017', fold=0, outpath='FINENNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='017', fold=0, outpath='FINENNUNET', val=True, npz=True, na=True)


# Livus
run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='130', fold=0, outpath='FINENNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='130', fold=0, outpath='FINENNUNET', val=True, npz=True, na=True)



# WORD
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='140', fold=0, outpath='FINENNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='140', fold=0, outpath='FINENNUNET', val=True, npz=True, na=True)

