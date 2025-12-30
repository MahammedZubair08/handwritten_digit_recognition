# ✍️ Handwritten Digit Recognition

A Machine Learning–based web application that recognizes handwritten digits from images using a Convolutional Neural Network (CNN). The application supports **single and multiple digit recognition** and is deployed with an interactive **Streamlit** interface.

---

## 📌 Project Overview

Handwritten digit recognition is a classic computer vision problem with real-world applications such as:
* Automated form processing  
* Cheque and document digitization  
* Optical Character Recognition (OCR) systems  

This project uses a **pre-trained CNN model on the MNIST dataset** and applies robust image preprocessing techniques to accurately predict digits from uploaded handwritten images.



---

## 🚀 Features

* 🧠 **CNN-based digit classification** * 🖼️ **Upload handwritten digit images** * 🔢 **Supports multi-digit recognition** * ⚙️ **Advanced image preprocessing:**
    * Grayscale conversion  
    * Gaussian blur  
    * Adaptive / Otsu thresholding  
    * Contour detection & segmentation  
* 📱 **Responsive UI** for desktop and mobile  
* 🧪 **Debug mode** to visualize preprocessing steps  

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python |
| **ML Framework** | TensorFlow / Keras |
| **Model** | Convolutional Neural Network (CNN) |
| **Dataset** | MNIST |
| **Image Processing** | OpenCV, PIL |
| **Web Framework** | Streamlit |
| **Deployment** | Local / Cloud (Streamlit) |

---

## 🧠 Model Details

* **Input Shape:** $28 \times 28$ grayscale images  
* **Architecture:** * Convolution Layers  
    * MaxPooling  
    * Fully Connected Dense Layers  
* **Optimizer:** Adam  
* **Loss Function:** Categorical Crossentropy  
* **Accuracy:** ~99% on MNIST test data  

---

## 🖼️ Image Preprocessing Pipeline

1.  Convert image to grayscale  
2.  Apply Gaussian blur to reduce noise  
3.  Apply thresholding for digit isolation  
4.  Detect contours of individual digits  
5.  Sort digits from left to right  
6.  Resize each digit to $28 \times 28$  
7.  Normalize and predict using CNN  

---

## 📂 Project Structure

```yml
handwritten_digit_recognition/
│
├── app.py                 # Streamlit application
├── DigitClassifier.keras  # Trained CNN model
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── sample_images          # Test image  
```
---

## ▶️ How to Run the Project

1️⃣ Clone the Repository
---
```
Bash

git clone https://github.com/MahammedZubair08/handwritten_digit_recognition.git


cd handwritten_digit_recognition
```
2️⃣ Install Dependencies
---
```
Bash

pip install -r requirements.txt
```
## 3️⃣ Run the Streamlit App
```
Bash
streamlit run app.py
```
---
4️⃣ Open in Browser
---
```

http://localhost:8501
```