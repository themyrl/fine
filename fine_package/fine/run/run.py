import run_training as run
import sys
import argparse


g = '0'
# ALL
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-network", type=str)
	parser.add_argument("-task", type=str)
	parser.add_argument("-outpath", type=str)
	parser.add_argument("-na", action="store_true")
	parser.add_argument("-only_val", action="store_true")
	parser.add_argument("-tta", action="store_true")
	parser.add_argument("-continu", action="store_true")
	parser.add_argument("-clip", action="store_true")

	args = parser.parse_args()



	# network_trainer = sys.argv[1]
	# task = str(sys.argv[2])
	# outpath = sys.argv[3]
	# na = bool(int(sys.argv[4]))
	# ov = bool(int(sys.argv[5]))
	# tta = bool(int(sys.argv[6]))

	# if len(sys.argv) > 7:
	# 	c = bool(int(sys.argv[7]))
	# else:
	# 	c = False

	network_trainer = args.network
	task = args.task
	outpath = args.outpath
	na = args.na
	ov = args.only_val
	tta = args.tta
	c = args.continu
	clip = args.clip


	if not ov:
		if clip:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=False, npz=True, na=na, tta=tta, c=c, clip=clip)
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, clip=clip)
		else:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=False, npz=True, na=na, tta=tta, c=c)
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta)
	else:
		if clip:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, clip=clip)
		else:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta)
