# Embryo-Image-Classifier

This project focuses on classifying embryo images into different developmental stages—such as 8-cell, Morula, and Blastocyst—using deep learning techniques. The model is trained on labeled microscope images and uses Convolutional Neural Networks (CNNs) for accurate stage identification, which can aid in improving decision-making in IVF procedures.

---

# Objective

To automate the classification of embryo images and support clinical decision-making in IVF procedures by accurately predicting the developmental stage and quality grade of embryos.

---

# Model Summary

- **Input:** Microscopic embryo images  
- **Output:**  
  - Predicted stage – `8-cell`, `Morula`, `Blastocyst`  
  - Predicted grade – `A`, `B`, `C`  
- **Architecture:** Pre-trained DenseNet201 (fine-tuned)  
- **Framework:** TensorFlow / Keras  

---

# Dataset

- The dataset consists of labeled embryo images categorized into:
  - `8Cell_A`, `8Cell_B`, `8Cell_C`
  - `Morula_A`, `Morula_B`, `Morula_C`
  - `Blastocyst_A`, `Blastocyst_B`, `Blastocyst_C`

- **Data Augmentation** techniques (rotation, flipping, zooming, etc.) are used to handle class imbalance and enhance generalization.

---

# Installation

1. **Clone the repository**
```bash
git clone https://github.com/jeev2727/Embryo-Image-Classifier.git
cd Embryo-Image-Classifier


