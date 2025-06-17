import streamlit as st
import joblib
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_model():
    model = joblib.load('resume_model.joblib')
    vectorizer = joblib.load('resume_vectorizer.joblib')
    return model, vectorizer

def predict_job_name(resume_text, model, vectorizer):
    resume_clean = clean_text(resume_text)
    resume_vec = vectorizer.transform([resume_clean])
    return model.predict(resume_vec)[0]

def main():
    st.title("Resume Job Matcher")
    st.write("Upload your resume text to predict the most suitable job role.")
    
    # Load model
    try:
        with st.spinner('Loading model...'):
            model, vectorizer = load_model()
        
        # Input area
        resume_text = st.text_area("Paste your resume text here:", height=300)
        
        if st.button("Predict Job Role"):
            if resume_text:
                with st.spinner('Analyzing resume...'):
                    predicted_job = predict_job_name(resume_text, model, vectorizer)
                    st.success(f"Predicted Job Role: {predicted_job}")
            else:
                st.warning("Please enter some resume text.")
    except FileNotFoundError:
        st.error("Model files not found. Please run matcher.py first to train and save the model.")

if __name__ == "__main__":
    main()