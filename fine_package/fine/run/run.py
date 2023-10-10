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
	parser.add_argument("-nodeter", action="store_true")
	parser.add_argument("-visu", action="store_true")
	parser.add_argument("-vb", action="store_true")
	parser.add_argument("-idx", type=int, default=-1)
	parser.add_argument("-ntok", type=int, default=1)
	parser.add_argument("-depths", nargs='+', type=int, default=[2])
	parser.add_argument("-dofine", nargs='+', type=int, default=[0,0,1,1,1,1])
	parser.add_argument("-ep", type=float, default=1000)
	parser.add_argument("-lr", type=float, default=1e-2)
	parser.add_argument("-prett", type=str, default="")

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
	idx = args.idx
	deterministic = not args.nodeter
	ntok = args.ntok
	visu = args.visu
	depths = args.depths
	vb = args.vb
	ep = args.ep
	lr = args.lr
	prett = args.prett
	if len(depths) == 1:
		depths = [depths[0] for i in range(6)]
	elif len(depths) < 6:
		print("error len of depths")
		exit(0)
	dofine = args.dofine
	if len(depths) < 6:
		print("error len of dofine")
		exit(0)

	if not ov:
		if ntok != 1 or depths != [2, 2, 2, 2, 2, 2]  or dofine!=[0,0,1,1,1,1]:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=False, npz=True, na=na, tta=tta, c=c, deterministic=deterministic, 
				vt_num=ntok, depths=depths, dofine=dofine, ep=ep, lr=lr, pretrained_weights=prett)
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, deterministic=deterministic, 
				vt_num=ntok, depths=depths, dofine=dofine, ep=ep, lr=lr, pretrained_weights=prett)
		# if clip:
		# 	run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=False, npz=True, na=na, tta=tta, c=c, clip=clip)
		# 	run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, clip=clip)
		else:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=False, npz=True, na=na, tta=tta, c=c, deterministic=deterministic, ep=ep, lr=lr, pretrained_weights=prett)
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, deterministic=deterministic, ep=ep, lr=lr, pretrained_weights=prett)
	else:
		if ntok != 1 or depths != [2, 2, 2, 2, 2, 2]  or dofine!=[0,0,1,1,1,1]:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, deterministic=deterministic, 
				vt_num=ntok, depths=depths, dofine=dofine, visu=visu, vb=vb)
		# if clip:
		# 	run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, clip=clip)
		else:
			run.main(gpu=g, network='3d_fullres', network_trainer=network_trainer, task=task, fold=0, outpath=outpath, val=True, npz=True, na=na, tta=tta, deterministic=deterministic, visu=visu, idx=idx, vb=vb)
