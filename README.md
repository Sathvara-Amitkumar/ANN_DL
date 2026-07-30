# 🌸 Iris Flower Classification using Artificial Neural Network (ANN)

A Deep Learning web application built with **TensorFlow/Keras** and **Streamlit** that predicts the species of an Iris flower based on four flower measurements.
---

## 📌 Project Overview

This project uses an **Artificial Neural Network (ANN)** trained on the famous **Iris Dataset** to classify flowers into one of three species:

- 🌸 Iris Setosa
- 🌼 Iris Versicolor
- 🌺 Iris Virginica

The trained model is deployed using **Streamlit Community Cloud**, allowing users to enter flower measurements and receive instant predictions with confidence scores.

---

## 🧠 Model Information

| Property | Value |
|----------|--------|
| Model | Artificial Neural Network (ANN) |
| Framework | TensorFlow / Keras |
| Dataset | Iris Dataset |
| Test Accuracy | **96%** |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Activation Functions | ReLU, Softmax |

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```text
ANN_DL/
│
├── app.py
├── iris_model.h5
├── scaler.pkl
├── encoder.pkl
├── requirements.txt
├── README.md
└── iris_prediction.ipynb
```
---

## ▶️ Run Locally

Clone the repository

```bash
git clone https://github.com/Sathvara-Amitkumar/ANN_DL.git
```

Move into the project

```bash
cd ANN_DL
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```
