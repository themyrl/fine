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
@misc{https://doi.org/10.48550/arxiv.2210.05313,
  doi = {10.48550/ARXIV.2210.05313},
  
  url = {https://arxiv.org/abs/2210.05313},
  
  author = {Themyr, Loic and Rambour, Clément and Thome, Nicolas and Collins, Toby and Hostettler, Alexandre},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences, 68T45},
  
  title = {Memory transformers for full context and high-resolution 3D Medical Segmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## 5. Acknowledgements

Part of codes are reused from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Fabian Isensee for the codes of nnU-Net.



## Contact
Loic THEMYR ([loic.themyr@lecnam.net](loic.themyr@lecnam.net))
