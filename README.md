# 🎯 Resume Job Matcher

This is a simple machine learning web app built with **Python** and **Streamlit** that predicts the most suitable job role based on a given resume text. It uses **Logistic Regression** trained on a small dataset of resumes.

---

## 📌 Features

- Predicts job roles using resume text
- Preprocessing with regex and TF-IDF vectorization
- Multinomial Logistic Regression model
- Simple and interactive frontend using Streamlit

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML**: scikit-learn (Logistic Regression)
- **NLP**: TfidfVectorizer
- **Other Libraries**: pandas, numpy, regex

---

## 📊 Model Info

- Algorithm: `LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')`
- Trained on: Small resume dataset (`UpdatedResumeDataSet.csv`)
- Vectorizer: `TfidfVectorizer(max_features=5000, stop_words='english')`

> ⚠️ Since the model was trained on a **very small dataset**, it may not always return perfect predictions. This project is intended for **educational/demo purposes** only.

---

## 🚀 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Lakhan-Bhutiya/resume-job-matcher.git
    cd resume-job-matcher
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Example

```txt
Input Resume: 
"Skilled in Python, TensorFlow, deep learning, and model optimization..."

Predicted Job Role:
"Data Scientist"

---

## 🌐 Live Demo

🔗 [Click here to try the Resume Job Matcher]([[https://<your-app-name>.streamlit.app](https://resume-job-matcher13.streamlit.app/)](https://resume-job-matcher13.streamlit.app/))

---

## 🧠 Notes

- This model was trained on a **small dataset** for demonstration purposes.
- Results might not be accurate for all types of resumes.
- You can improve it by using a larger, cleaned dataset and experimenting with other models like SVM or fine-tuned Transformers (BERT).

---

