# Welcome to our AI 2 Computer Vision Project!

## 💡 About
 📌 This is a deep learning project consisting of three members whose primary focus is to detect benign brain tumors using a multi-classification computer vision (CV) approach. The scope focuses on three classes: meningioma, glioma, and pituitary tumors from the Kaggle dataset: https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification/data. First, the goal of this project is to contribute artificial intelligence tasks in medical fields such as tumor detection. Second, it is also to benchmark newer models such as the YOLOv12, a powerful computer vision model that uses attention-based modules such as FlashAttention and area attention against complex visual environments. 

 ## 👥 Contributors

| Contributors | Name                          | GitHub Username | Role/Responsibility                                                                 | LinkedIn |
|--------------|-------------------------------|-----------------|-------------------------------------------------------------------------------------|----------|
| Member #1    | Aaron Gabriel L. Novesteras   | [@Paradoxidus](https://github.com/Paradoxidus) | Paper writing, training & validation, testing, results analysis                     | [LinkedIn](https://www.linkedin.com/in/aaron-gabriel-novesteras-077a352a9) |
| Member #2    | Gabriel Christian D. Tan      | -             | Documentation (Colab notebooks, paper, supporting materials)                        | - |
| Member #3    | Thomas Kaden K. Zerda         | -             | Training, paper writing, deployment                                                 | - |

## 📦 Dependencies

This project requires the following libraries and tools to run YOLOv12 for brain tumor detection:

- **Core Frameworks**
  - torch==2.2.2  
  - torchvision==0.17.2  
  - ultralytics==8.3.176  

- **Computer Vision & Augmentation**
  - timm==1.0.14  
  - albumentations==2.0.4  
  - opencv-python==4.9.0.80  
  - pycocotools==2.0.7  

- **Utilities**
  - PyYAML==6.0.1  
  - scipy==1.13.0  
  - numpy==1.26.4  
  - ps

## 📊 Results and Summary

| Tumor Class | AP @ IoU=0.5 | AP @ IoU=0.5–0.95 | Precision | Recall | F1 (Peak) |
|-------------|--------------|-------------------|-----------|--------|-----------|
| Glioma      | 0.754        | ~0.45             | ~0.71     | ~0.70  | 0.71 @ 0.30–0.35 |
| Meningioma  | 0.982        | ~0.55             | >0.95     | >0.90  | >0.90 (stable across 0.05–0.70) |
| Pituitary   | 0.802        | ~0.50             | ~0.80     | ~0.80  | 0.72 @ 0.30–0.35 |
| **Overall** | 0.846        | ~0.54             | >0.80     | ~0.75  | 0.80 @ 0.304 |



 For additional infomation including as graphs, model architecture, and among others, you may refer to the group's IEEE published conference paper: [10.1109/ICISS67859.2026.11453982.](https://doi.org/10.1109/ICISS67859.2026.11453982). It is also provided in the repository along with the deployment, code, etc.

 Core architecture/template of YOLOv12 for transfer learning was created and trained by [Dr. Lysa V. Comia.](https://ieeexplore.ieee.org/author/237244627179273).

 
