# 🎬 Movie Review Sentiment Analysis with Simple RNN 🎭  

## 🚀 Overview  
This project implements a **Sentiment Analysis** model using a **Simple Recurrent Neural Network (RNN)** to classify movie reviews as **positive or negative**. The model is trained on textual movie reviews and deployed as a **live web app** for real-time predictions. 

## 💻 Live Demo  
🔗 **Try the Live App**: [Movie Review Sentiment Classifier](https://moviereviewsentimentanalysisrnn-ma.streamlit.app/)  

## 🏆 Features  
✅ Train a **Simple RNN** model for sentiment classification  
✅ Preprocess movie reviews (padding)  
✅ Use **IMDb dataset**
✅ Deploy a **live web app** for real-time sentiment prediction  
✅ Interactive UI using **Streamlit** 


## 📊 Dataset  
The dataset used is the **IMDb Movie Reviews Dataset** from TensorFlow/Keras datasets.  
- Contains **50,000** labeled movie reviews  
- Labels: **Positive (1) / Negative (0)**  
- Text data needs preprocessing (tokenization, padding, embeddings)  

## 🏗️ Model Architecture  
- **Embedding Layer**: Converts words into dense vector representations  
- **Simple RNN Layer**: Captures sequential dependencies  
- **Dense Layer**: Fully connected layer for classification  
- **Output Layer**: Uses **Sigmoid activation** for binary classification  

## 🎯 Training Pipeline  
1. **Data Preprocessing**: Tokenization, padding, text vectorization  
2. **Model Training**: Train **Simple RNN** model on preprocessed data  
3. **Evaluation**: Check accuracy, confusion matrix, F1-score  
4. **Deployment**: Save model and integrate it into the web app  




 
