import numpy as np
import argparse
import os
import json

from medpy.metric.binary import assd
import nibabel as nib


all_tasks = {"017":"BCV",
			"130":"Livus",
			"140":"WORD",
			}



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-path", type=str, default="/gpfsstore/rech/arf/unm89rb/nnUNet_trained_models/nnUNet/3d_fullres_nnUNetPlansv2.1/")
	parser.add_argument("-network", type=str)
	parser.add_argument("-task", type=str)
	parser.add_argument("-pp", action="store_true")
	parser.add_argument("-n_classe", type=int)
	# parser.add_argument("-outpath", type=str)


	args = parser.parse_args()

	n_classe = args.n_classe
	path = args.path
	all_tasks = os.lisdir(path)
	for task in all_tasks:
		all_network = os.lisdir(os.join(path, task))
		for network in all_network:
			pps = ["fold_0/validation_raw", "fold_0/validation_raw_postprocessed"]
			p = 0
			for pp in pps:
				try:
					# network = "{}_IN_LeakyReLU/".format(args.network)
					# task = "Task{}_{}/".format(args.task, all_tasks[args.task])

					# pred = "fold_0/validation_raw"
					# if args.pp:
						# pred += "_postprocessed"
					# pred += "/"


					target_fold = os.path.join(path, task, network, "gt_niftis/")
					pred_fold = os.path.join(path, task, network, pp)
					outpath = os.path.join(path, task, network, "assd_results_{}.json".format(p))

					print("------------------------------------------------------------------------------")
					print(pred_fold)


					results = {}
					res_per_classes_ = {}
					for i in range(1, n_classe+1):
						res_per_classes_[str(i)] = []

					# compute assd per patient
					for i in os.listdir(pred_fold):
						if "nii.gz" in i:
							patient_id = i.split(".")[0]
							print(patient_id)

							res = {}

							target = nib.load(os.path.join(target_fold, i)).get_fdata()
							pred = nib.load(os.path.join(pred_fold, i)).get_fdata()

							# print()
							# print(np.unique(target))
							# print(np.unique(pred))

							for j in range(1, n_classe+1):
								res[str(j)] = assd(1*(pred == j), 1*(target == j))
								res_per_classes_[str(j)].append(res[str(j)])
								print(res[str(j)], end=" ")

							results[patient_id] = res
							print("\n\n")

					# compute mean and std
					alls = {}
					means = []
					stds = []
					for i in range(1, n_classe+1):
						alls[str(i)] = {"mean": np.mean(res_per_classes_[str(i)]),
										"std": np.std(res_per_classes_[str(i)])}
						means.append(alls[str(i)]["mean"])
						stds.append(alls[str(i)]["std"])

					alls["mean_all"] = {"mean": np.mean(means),
										"std": np.mean(stds)}

					print(alls)

					results["all"] = alls

					with open(outpath, 'w', encoding='utf-8') as f:
						json.dump(results, f, ensure_ascii=False, indent=4)
				except:
					pass










