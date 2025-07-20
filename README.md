# Deepfake Image Detection 🎭🧠

This project is a deep learning-based web application for detecting deepfake images using a fine-tuned **Xception** model. The app allows users to upload an image and predicts whether it is **Real** or **Fake**.
Try with examples given in the repo 
Live Link:- https://image-verifier.streamlit.app/

## 🔍 Overview

- **Dataset**: [Deepfake and Real Images by Manjil Karki](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- **Model**: Xception CNN (pretrained on ImageNet, fine-tuned on the dataset)
- **Frontend**: [Streamlit](https://streamlit.io/)
  

## 🚀 Features

- 🔄 Drag & drop image upload
- ✅ Real-time prediction: Fake / Real



## 🧠 Model Architecture

- **Base Model**: Xception
- **Evaluation Metric**: Accuracy
- **Input Size**: 299x299x3

The model is trained on labeled real and fake images and saved as a `.h5` file for deployment in the Streamlit web app.

## 📁 Project Structure
Deepfake_detection/ │ ├── streamlit_app.py # Main Streamlit app ├── model/ │ └── xception_model.h5 # Trained model ├── images/ │ └── sample_images/ # Sample test images ├── requirements.txt # Python dependencies └── README.md 

# Project description

## 🛠️ Installation & Run

1. **Clone the repo and run the project**
   ```bash
   git clone https://github.com/Aafimalek/Deepfake_detection
   cd Deepfake_detection
   pip install -r requirements.txt
   streamlit run streamlit_app.py


