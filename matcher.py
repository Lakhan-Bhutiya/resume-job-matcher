import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
import string
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Resume', 'Job Name'])
    df['Resume_clean'] = df['Resume'].apply(clean_text)
    return df

def train_model(df):
    X = df['Resume_clean']
    y = df['Job Name']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    joblib.dump(model, 'resume_model.joblib')
    joblib.dump(vectorizer, 'resume_vectorizer.joblib')
    
    return model, vectorizer

def predict_job_name(resume_text, model, vectorizer):
    resume_clean = clean_text(resume_text)
    resume_vec = vectorizer.transform([resume_clean])
    return model.predict(resume_vec)[0]

def main():
    df = load_and_preprocess('UpdatedResumeDataSet.csv')
    model, vectorizer = train_model(df)
    # Example prediction
    sample_resume = df.iloc[0]['Resume']
    predicted_job = predict_job_name(sample_resume, model, vectorizer)
    print(f'Predicted job for sample resume: {predicted_job}')

if __name__ == '__main__':
    main()