# Brain Tumor Auto-Segmentation with U-Net  
*A Quantitative Comparison of Focal Loss and Binary Cross-Entropy Loss*

[![Paper DOI](https://img.shields.io/badge/DOI-10.18502/jbe.v11i1.19315-blue)](https://doi.org/10.18502/jbe.v11i1.19315)

Official implementation of the paper:  
**A Quantitative Comparison between Focal Loss and Binary Cross-Entropy Loss in Brain Tumor Auto-Segmentation Using U-Net**  
Published in *Journal of Biostatistics and Epidemiology*, 2025 

---

### Overview
Brain tumor segmentation from MRI scans is a critical task for early diagnosis, treatment planning, and outcome monitoring. This repository provides the code and workflows to reproduce our study comparing **Focal Loss** and **Binary Cross-Entropy (BCE) Loss** in training a U-Net model for automatic tumor segmentation.

Our experiments show that **Focal Loss outperforms BCE Loss** in terms of Dice coefficient, precision, recall, and overall segmentation robustness on imbalanced datasets.

---

### Contributions
- Built and trained **U-Net architecture** with 4 encoding and 4 decoding blocks (*see Figure 1, p.6*).  
- Curated a **dataset of 314 high-resolution brain MRI scans** (sagittal, coronal, axial planes) from 108 patients.  
- Implemented **skull stripping** preprocessing for noise reduction and improved localization (*see Figure 3, p.8*).  
- Compared **loss functions** (BCE vs. Focal Loss) with extensive **hyperparameter tuning** via Ray Tune.  
- Reported evaluation using **Dice (F1), Precision, Recall, Accuracy, and Hausdorff Distance**.  

---

### Results
5-fold cross-validation summary (test data):

| Loss Function | Accuracy | Dice (F1) | Precision | Recall |
|---------------|----------|-----------|-----------|--------|
| **Focal Loss** | 99.44%   | **81%**   | **82.92%** | **79.32%** |
| BCE Loss      | 99.03%   | 74.52%    | 76.16%    | 71.9%  |

- Focal Loss improved segmentation by **+6.4% Dice**, **+6.8% Precision**, **+7.4% Recall** compared to BCE.  
- Hausdorff distance with Focal Loss: **95% CI = (43.22mm, 52.92mm)** (*Table 3, p.14*).  

---

### Dataset
- **Source**: MRI scans collected at Bahar Medical Imaging Center (2021–2022).  
- **Size**: 314 images (800×512 pixels) with expert-annotated masks (*examples in Figure 2, p.7*).  
- **Preprocessing**: Skull stripping, Gaussian blurring, Otsu’s thresholding; multi-plane (sagittal, axial, coronal).  


---

### Model Architecture
Here is the structure of model

![model](images/Model.png)

---

#### ⚙️ Implementation Details
- **Frameworks**: Python 3.9, TensorFlow 2.8, Keras  
- **Environment**: Google Colab TPU acceleration  
- **IDE**: Spyder 5.0.1  
- **Validation**: 5-fold cross-validation  

---

### Installation
``` bash
git clone https://github.com/mahdishafiei/Brain-Tomur-Semantic-segmentation.git
cd Brain-Tomur-Semantic-segmentation
pip install -r requirements.txt
```

### citation

If you use this code in your research, please cite our paper:

```bibtex
@article{neyestanak2025quantitative,
  title={A quantitative comparison between focal loss and binary cross-entropy loss in brain 
  tumor auto-segmentation using U-Net},
  author={Neyestanak, Mahdi Shafiei and Jahani, Hamid and Khodarahmi, Mohsen and Zahiri, Javad and Hosseini, Mostafa and Fatoorchi, Amirali and Yekaninejad, Mir Saeed},
  journal={Journal of Biostatistics and Epidemiology},
  volume={11},
  number={1},
  pages={15--35},
  year={2025}
}
```
