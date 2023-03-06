import run_training as run
import sys


g = '0'
# ALL
if __name__ == '__main__':
	network_trainer = sys.argv[1]
	task = str(sys.argv[2])
	outpath = sys.argv[3]
	na = bool(int(sys.argv[4]))
	ov = bool(int(sys.argv[5]))
	tta = bool(int(sys.argv[6]))

	if len(sys.argv) > 7:
		c = bool(int(sys.argv[7]))
	else:
		c = False

	if not ov:
		run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=False, npz=True, na=na, tta=tta, c=c)
		run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta)
	else:
		run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta)
