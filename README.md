# NeuroInsight: AI-Powered Brain Tumor Diagnosis & Segmentation

Welcome to **NeuroInsight**, an innovative AI-driven platform designed to revolutionize the detection, segmentation, and classification of brain tumors using state-of-the-art deep learning techniques. This project leverages cutting-edge technologies and advanced deep neural networks to automate critical aspects of brain cancer diagnosis, providing a more accurate, efficient, and reliable alternative to traditional methods.

---

## 🚀 **Project Overview**
Brain tumor detection and diagnosis can be a time-consuming and error-prone process when relying on manual methods. **NeuroInsight** aims to address these challenges by integrating AI to:
- **Detect the presence of brain tumors** in MRI scans.
- **Segment tumor regions** into key sub-regions: edema, enhancing tumor, and non-enhancing tumor.
- **Classify tumor types** into **glioblastoma** or **lower-grade glioma**.

By automating these tasks, **NeuroInsight** significantly reduces human error and the time required for diagnosis, empowering healthcare professionals with a more reliable tool for accurate decision-making.

![ct-scan-brain-doctors-hand-600nw-2496120619](https://github.com/user-attachments/assets/1fa7b52c-7914-40ed-9dc4-6070b55eea59)

## 🧠 **Why NeuroInsight?**
- **Accurate Diagnosis**: By leveraging deep learning models (U-Net, TensorFlow, Keras), NeuroInsight ensures high precision in tumor detection and segmentation.
- **Enhanced Efficiency**: The platform automates tedious tasks, drastically reducing the time doctors and radiologists spend on image analysis.
- **Cutting-Edge AI**: Built on TensorFlow and Keras, the project showcases the power of modern neural networks in solving complex healthcare challenges.
- **Comprehensive Solution**: From detection to segmentation and classification, NeuroInsight provides an all-in-one solution for brain tumor analysis.

---

## 💡 **Technologies Used**
- **Python**: The backbone of the project, ensuring flexibility and ease of integration.
- **TensorFlow & Keras**: State-of-the-art deep learning frameworks to train, fine-tune, and deploy neural network models.
- **U-Net**: A highly effective architecture for image segmentation tasks, used to segment tumor regions accurately.
- **BraTS Dataset**: A curated collection of multi-parametric MRI scans with expert annotations, providing a high-quality dataset for model training and validation.

![1711127950444](https://github.com/user-attachments/assets/5474390e-f6ee-454c-9017-5bd34e4ef453)

---

## 🛠️ **How It Works**
1. **Data Preprocessing**: The MRI scans are preprocessed to ensure consistency, normalization, and preparation for model training.
2. **Tumor Detection**: AI models are trained to predict the presence of tumors in the provided MRI scans.
3. **Tumor Segmentation**: The model performs pixel-level segmentation to classify tumor sub-regions (edema, enhancing, non-enhancing tumor).
4. **Tumor Classification**: Using the segmented data, the model classifies tumor types, distinguishing between glioblastoma and lower-grade gliomas.
5. **Web Application**: A user-friendly web app that allows healthcare professionals to upload MRI scans and receive automated reports, with interactive visualizations of tumor regions.

![Step 5- Model Processing](https://github.com/user-attachments/assets/853e3e55-980f-4841-b803-a71faf467e3f)

---

## 👨‍💻 **Getting Started**
Clone the repository, set up your environment, and follow the steps in the documentation to run the application or use the models for tumor analysis.

```bash
git clone https://github.com/yourusername/NeuroInsight.git
cd NeuroInsight
pip install -r requirements.txt
