**Overview**
This project implements a real-time emotion detection system using facial recognition. By leveraging computer vision and deep learning techniques, the system classifies facial expressions into distinct emotional categories such as happiness, sadness, anger, surprise, and more. 
The project is built using a Convolutional Neural Network (CNN) model trained on a publicly available dataset. This version also features a Q/A related to mental health, the data from the Q/A is saved to an excel file.

**Key Features**
Real-Time Emotion Detection: Analyze emotions from live camera feeds or static images.
Accurate Classification: High precision due to advanced CNN architecture.
Non-Intrusive Methodology: Facial recognition as the sole input format.
Applications: Mental health monitoring, customer service enhancement, and human-computer interaction.


**Programming Language:** Python

**Libraries and Frameworks:**

Tkinter

TensorFlow

Keras

OpenCV

NumPy

**Dataset:** FER-2013 (Facial Expression Recognition dataset)

**Methodology**

  Data Preprocessing:
      Resize and normalize facial images.
      Apply data augmentation to improve model generalization.
      
  Model Training:
      Train a CNN on the FER-2013 dataset for emotion classification.
      Optimize the model for high accuracy across diverse facial expressions.
      
  Real-Time Detection:
      Use OpenCV for face detection from video streams.
      Classify emotions using the trained CNN model.
