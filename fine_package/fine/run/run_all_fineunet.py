# import UTrans.run.nnunet_run_training as run
import run_training as run
import argparse


g = '0'


# BCV
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='017', fold=0, outpath='FINENNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='017', fold=0, outpath='FINENNUNET', val=True, npz=True, na=True)


# Livus
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='130', fold=0, outpath='FINENNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='130', fold=0, outpath='FINENNUNET', val=True, npz=True, na=True)



# WORD
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='140', fold=0, outpath='FINENNUNET', val=False, npz=True, na=True)
# run.main(gpu=g, network='3d_fullres', network_trainer='nnUNetTrainerV2_fineUNet', task='140', fold=0, outpath='FINENNUNET', val=True, npz=True, na=True)

# ALL
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--network_trainer")
	parser.add_argument("-t", "--task", type=str)
	parser.add_argument("-o", "--outpath")
	parser.add_argument("-a", "--na", type=bool)

	args = parser.parse_args()

	run.main(gpu=g, network='3d_fullres', network_trainer=args['network_trainer'], task=args['task'], fold=0, outpath=args['outpath'], val=False, npz=True, na=args['na'])
	run.main(gpu=g, network='3d_fullres', network_trainer=args['network_trainer'], task=args['task'], fold=0, outpath=args['outpath'], val=True, npz=True, na=args['na'])
