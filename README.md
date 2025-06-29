# Neuro Insight 🧠

A medical web application powered by AI for automated analysis of brain MRI scans, specifically designed to detect and classify glioma brain tumors.
![image](https://github.com/user-attachments/assets/771dd969-0146-4bd9-b3ff-f66fd00661c5)
![image](https://github.com/user-attachments/assets/032caca0-f5ce-4894-9fd1-c7aa740dc8e4)
![image](https://github.com/user-attachments/assets/2a578f1f-859c-46be-9f85-dfb0a152880b)

![screencapture-127-0-0-1-8000-home-screen-2025-06-03-06_41_43](https://github.com/user-attachments/assets/b5bef42c-a2c0-4512-9424-954de9ed7191)
![mockuper (1)](https://github.com/user-attachments/assets/c0b9cb82-ef48-4255-bebf-52976885f278)

## ✨ Key Features

- **🔍 Automated Tumor Detection**: Instantly identifies presence of brain tumors in MRI scans
- **📊 Tumor Segmentation**: Precisely segments tumor regions using Attention U-Net architecture
- **🏷️ Grade Classification**: Classifies tumors as HGG (aggressive) or LGG (less aggressive) using ConvNeXt
- **📄 Report Generation**: Creates comprehensive PDF reports for clinical use
- **👤 User Management**: Secure authentication and profile management for medical professionals
- **💾 Report Storage**: Stores and retrieves patient reports with database integration

## 🏗️ Architecture

- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Backend**: Django (Python)
- **AI Models**: 
  - Attention U-Net for tumor segmentation
  - ConvNeXt-Base for grade classification
- **Database**: SQLite3
- **Dataset**: BraTS 2019

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/UsmanK7/Neuro-Insight.git
   cd neuro-insight
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up database**
   ```bash
   python manage.py migrate
   ```

5. **Run the application**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:8000`

## 📋 Usage

1. **Sign Up/Login**: Create an account or login as a medical professional
2. **Upload MRI Scans**: Upload FLAIR and T1CE modalities with slice selection
3. **AI Analysis**: The system automatically:
   - Detects tumor presence
   - Segments tumor regions if present
   - Classifies tumor grade (HGG/LGG)
4. **View Results**: Review segmented images and classification results
5. **Generate Reports**: Create and download comprehensive diagnostic reports
6. **Manage Reports**: Access saved reports from your profile

## 🔬 Model Performance

### Segmentation Model (Attention U-Net)
- **Accuracy**: 99.30%
- **Dice Coefficient**: 99.47%
- **Mean IoU**: 50.00%
- **Precision**: 99.38%
- **Sensitivity**: 99.24%

### Classification Model (ConvNeXt-Base)
- **Accuracy**: 98.66%
- **AUC**: 99.50%
- **Sensitivity**: 99.54%
- **Precision**: 94.30%
- **F1-Score**: 99.20%

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Web Framework** | Django |
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript |
| **Deep Learning** | TensorFlow, Keras |
| **Image Processing** | OpenCV, nibabel |
| **Data Science** | NumPy, pandas, scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Database** | SQLite3 |

## 📊 Dataset

- **Source**: BraTS 2019 (Brain Tumor Segmentation Challenge)
- **MRI Sequences**: T1CE and FLAIR modalities
- **Segmentation Labels**: 4 classes (background, necrotic core, edema, enhancing tumor)
- **Classification Labels**: HGG vs LGG

## 🔒 Security Features

- User authentication and session management
- Secure file upload validation
- Data encryption and privacy protection
- HIPAA-compliant data handling practices

## 🎯 Target Users

- **Primary**: Doctors and radiologists
- **Secondary**: Medical students and researchers
- **Requirements**: No technical AI/ML expertise needed

## 📈 Benefits

- ⚡ **Faster Diagnosis**: Reduces manual analysis time from hours to seconds
- 🎯 **Improved Accuracy**: AI-powered precision reduces human error
- 🌐 **Enhanced Accessibility**: Web-based platform for telemedicine
- 📚 **Educational Value**: Learning tool for medical professionals
- 💼 **Workflow Integration**: Seamless integration into clinical workflows

## 🔮 Future Enhancements

- [ ] Support for additional MRI modalities (T1, T2, T1-Gd)
- [ ] Expansion to other brain tumor types (meningioma, pituitary)
- [ ] Full 3D volume analysis
- [ ] Integration with hospital PACS systems
- [ ] Multi-language support
- [ ] Mobile application development

## ⭐ Acknowledgments

- BraTS 2019 dataset organizers
- University of Mianwali Computer Science Department
- Open source community for tools and libraries
- Medical professionals who provided domain expertise

---

**⚠️ Medical Disclaimer**: This application is for research and educational purposes. Always consult qualified medical professionals for clinical decisions. The AI predictions should be used as supplementary tools alongside professional medical judgment.
