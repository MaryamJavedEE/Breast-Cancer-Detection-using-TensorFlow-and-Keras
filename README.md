# Breast Cancer Detection Using Neural Networks

## Overview
This project aims to develop a breast cancer detection model using neural networks implemented with TensorFlow and Keras. The model is designed to classify whether a tumor is benign or malignant based on medical imaging data.


## Background
Breast cancer is one of the most common types of cancer among women worldwide. Early detection significantly increases the chances of successful treatment. This project leverages machine learning techniques to assist healthcare professionals in diagnosing breast cancer more effectively.

## Technologies Used
- **Python**
- **TensorFlow**: An open-source library for machine learning.
- **Keras**: A high-level neural networks API running on top of TensorFlow.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-Learn**: For data preprocessing and evaluation metrics.
Breast Cancer Detection Using Neural Networks
Overview This project aims to develop a breast cancer detection model using neural networks implemented with TensorFlow and Keras. The model is designed to classify whether a tumor is benign or malignant based on medical imaging data.
Table of Contents - Background - Technologies Used - Dataset - Installation - Usage - Model Architecture - Results - Contributing - License
Background Breast cancer is one of the most common types of cancer among women worldwide. Early detection significantly increases the chances of successful treatment. This project leverages machine learning techniques to assist healthcare professionals in diagnosing breast cancer more effectively.
Technologies Used - Python - TensorFlow: An open-source library for machine learning. - Keras: A high-level neural networks API running on top of TensorFlow. - Pandas: For data manipulation and analysis. - NumPy: For numerical computations. - Matplotlib/Seaborn: For data visualization. - Scikit-Learn: For data preprocessing and evaluation metrics.
Dataset The dataset used for this project is the Breast Cancer dataset, which is available from the scikit-learn library.  Key Features of the load_breast_cancer Dataset:
Purpose: The dataset is primarily used for the classification of tumors as either malignant (cancerous) or benign (non-cancerous) based on various attributes.

Data Structure: The dataset contains 30 features representing different characteristics of the tumors, such as:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Symmetry
Fractal Dimension
Target Variable: The target variable consists of binary labels:

0: Benign
1: Malignant
Number of Samples: The dataset contains 569 samples (tumor measurements).

Samples: 569 - Features: 30 numerical features, including radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry.
Installation To set up the project environment, follow these steps: 1. Clone the repository: bash git clone https://github.com/yourusername/breast-cancer-detection.git cd breast-cancer-detection 2. Install the required packages: bash pip install -r requirements.txt
Usage 1. Load the dataset and preprocess it: python import pandas as pd from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler # Load dataset data = pd.read_csv('data/breast_cancer_data.csv') 2. Train the model: python from keras.models import Sequential from keras.layers import Dense model = Sequential() model.add(Dense(32, activation='relu', input_shape=(30,))) model.add(Dense(16, activation='relu')) model.add(Dense(1, activation='sigmoid')) model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) model.fit(X_train, y_train, epochs=100, batch_size=10) 3. Evaluate the model: python loss, accuracy = model.evaluate(X_test, y_test) print(f"Accuracy: {accuracy * 100:.2f}%")
Model Architecture The neural network architecture consists of: - Input layer: 30 features - Hidden layers: - First hidden layer with 32 neurons and ReLU activation - Second hidden layer with 16 neurons and ReLU activation - Output layer: 1 neuron with sigmoid activation for binary classification
Results The model achieves an accuracy of approximately 96% on the test set . This performance demonstrates the potential of using neural networks for breast cancer detection.
Contributing Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any improvements.
