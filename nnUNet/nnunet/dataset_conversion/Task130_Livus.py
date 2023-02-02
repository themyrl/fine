#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == "__main__":
    base = "/gpfsscratch/rech/arf/unm89rb/LiverUS/"

    task_id = 130
    task_name = "Livus"
    prefix = 'LVU'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_folder = join(base, "imagesTr")
    label_folder = join(base, "labelStaple")
    test_folder = join(base, "")
    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        # serial_number = int(p[3:7])
        serial_number = int(p.split('_')[1])       

        # train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        train_patient_name = f'{prefix}_{serial_number:04d}.nii.gz'

        # label_file = join(label_folder, f'label{p[3:]}')
        label_file = join(label_folder, p.replace('_img', '-St_Vol'))

        image_file = join(train_folder, p)
        shutil.copy(image_file, join(imagestr, f'{train_patient_name[:7]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    # test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    # for p in test_patients:
    #     p = p[:-7]
    #     image_file = join(test_folder, p + ".nii.gz")
    #     serial_number = int(p[3:7])
    #     test_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
    #     shutil.copy(image_file, join(imagests, f'{test_patient_name[:7]}_0000.nii.gz'))
    #     test_patient_names.append(test_patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "Livus"
    json_dict['description'] = "Liver and vessels 3D ultrasound segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = ""
    json_dict['licence'] = "Ask authors"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "US",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "PV",
        "02": "IVC",
        "03": "HV",
        "04": "liver",
        }
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
