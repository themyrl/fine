# FINE : Memory transformers for full context and high-resolution 3D Medical Segmentation
[Memory transformers for full context and high-resolution 3D Medical Segmentation](https://arxiv.org/pdf/2210.05313.pdf) official repository

## 1. Installattion
```
conda create --name fine
conda activate fine
conda install pip

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd nnUNet
pip install -e .
cd ../fine_package
pip install -e .
```

## 2. Dataset

- Download [BCV Dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- Preprocess the BCV dataset according to the uploaded nnUNet package
- Training and Testing ID are in fine_package/fine/datasets/splits_final.pkl



## 3. Training & Evaluation

```
cd fine_package/fine
python run/run_all_fine.py
```


## 4. Citation
```
@InProceedings{10.1007/978-3-031-21014-3_13,
author="Themyr, Loic
and Rambour, Cl{\'e}ment
and Thome, Nicolas
and Collins, Toby
and Hostettler, Alexandre",
editor="Lian, Chunfeng
and Cao, Xiaohuan
and Rekik, Islem
and Xu, Xuanang
and Cui, Zhiming",
title="Memory Transformers for Full Context and High-Resolution 3D Medical Segmentation",
booktitle="Machine Learning in Medical Imaging",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="121--130",
abstract="Transformer models achieve state-of-the-art results for image segmentation. However, achieving long-range attention, necessary to capture global context, with high-resolution 3D images is a fundamental challenge. This paper introduces the Full resolutIoN mEmory (FINE) transformer to overcome this issue. The core idea behind FINE is to learn memory tokens to indirectly model full range interactions while scaling well in both memory and computational costs. FINE introduces memory tokens at two levels: the first one allows full interaction between voxels within local image regions (patches), the second one allows full interactions between all regions of the 3D volume. Combined, they allow full attention over high resolution images, e.g. 512{\$}{\$}{\backslash},{\backslash}times {\backslash},{\$}{\$}{\texttimes}512{\$}{\$}{\backslash},{\backslash}times {\backslash},{\$}{\$}{\texttimes}256 voxels and above. Experiments on the BCV image segmentation dataset shows better performances than state-of-the-art CNN and transformer baselines, highlighting the superiority of our full attention mechanism compared to recent transformer baselines, e.g. CoTr, and nnFormer.",
isbn="978-3-031-21014-3"
}
```

## 5. Acknowledgements

Part of codes are reused from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Fabian Isensee for the codes of nnU-Net.



## Contact
Loic THEMYR ([loic.themyr@lecnam.net](loic.themyr@lecnam.net))
