import streamlit as st
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(page_title="Fraud & Spam Detection System", page_icon="💡", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 44px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #1E90FF;
        font-weight: bold;
        margin-top: 10px;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        color: #696969;
        margin-top: 50px;
    }
    .expander-header {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_fraud_resources():
    # Load pre-trained resources for fraud detection
    with open("C:/Users/arjun/Assignment/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("C:/Users/arjun/Assignment/spam_text.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

@st.cache_resource
def load_spam_detection_resources():
    # Load LSTM model and tokenizer for website spam detection
    lstm_model = load_model("website_spam_link.h5")  # Replace with LSTM model path
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    return lstm_model, tokenizer

fraud_vectorizer, fraud_model = load_fraud_resources()
lstm_model, lstm_tokenizer = load_spam_detection_resources()

# Function for LSTM-based website spam detection
def predict_website_spam(url, model, tokenizer, max_length=200):
    if not url:
        return "Invalid input! The URL cannot be empty."

    # Tokenize and pad the URL
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts([url])
    url_sequence = tokenizer.texts_to_sequences([url])  # Pass URL as a list
    if not url_sequence or len(url_sequence[0]) == 0:
        return "URL contains no recognizable tokens!"

    # Pad the sequence
    padded_sequence = pad_sequences(url_sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict using the model
    prediction = model.predict(padded_sequence)
    return "Spam" if prediction[0][0] >= 0.5 else "Not Spam"
# Application title
st.markdown('<p class="main-header">💡 Fraud & Website Spam Detection System</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🌟 Navigation")
st.sidebar.info("""
Use this app to:
- Detect fraud in financial messages or files
- Identify whether a website URL is spam
""")

# Main introduction
st.write("---")
st.markdown("### 📚 Introduction")
st.info("""
Welcome to the **Fraud & Website Spam Detection System**!  
This tool uses advanced machine learning models to analyze:
1. Financial messages for potential fraud
2. Website URLs for spam activity  
Select a section below to begin.
""")

# Section 1: Financial Fraud Detection
with st.expander("💳 Financial Fraud Detection", expanded=True):
    st.markdown('<p class="sub-header">1️⃣ Financial Message Analysis</p>', unsafe_allow_html=True)
    # File upload analysis
    uploaded_file = st.file_uploader("Upload a text file containing financial messages (e.g., messages.txt):", type=["txt"])
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("📄 File Content Preview:", file_content, height=150)
        if st.button("🔍 Analyze Uploaded Messages"):
            with st.spinner("Analyzing the file..."):
                time.sleep(2)  # Simulate file analysis
                messages = [line.strip() for line in file_content.split("\n") if line.strip()]
                transformed_messages = fraud_vectorizer.transform(messages)
                predictions = fraud_model.predict(transformed_messages)
                fraud_count = (predictions == 'spam').sum()
            st.success(f"✔️ {fraud_count} fraudulent message(s) detected.")

    # Manual input analysis
    st.markdown('<p class="sub-header">2️⃣ Enter Financial Message</p>', unsafe_allow_html=True)
    user_message = st.text_area("Type or paste your financial message below:")
    if st.button("🔍 Analyze Manual Entry"):
        if user_message.strip():
            with st.spinner("Processing the message..."):
                time.sleep(1)
                transformed_message = fraud_vectorizer.transform([user_message])
                prediction = fraud_model.predict(transformed_message)
            if prediction[0] == 'spam':
                st.error("🚨 Fraud detected!")
            else:
                st.success("✔️ No fraud detected.")
        else:
            st.warning("⚠️ Please enter a valid message.")

# Section 2: Website Spam Detection
with st.expander("🌐 Website Spam Detection", expanded=True):
    st.markdown('<p class="sub-header">3️⃣ URL Spam Detection</p>', unsafe_allow_html=True)
    website_url = st.text_input("Enter the website URL for spam detection:")
    if st.button("🔍 Check URL"):
        if website_url.strip():
            with st.spinner("Analyzing the website..."):
                time.sleep(1)
                result = predict_website_spam(website_url, lstm_model, lstm_tokenizer)
            if result == "Spam":
                st.error(f"🚨 The URL '{website_url}' is classified as Spam.")
            else:
                st.success(f"✔️ The URL '{website_url}' is classified as Not Spam.")
        else:
            st.warning("⚠️ Please enter a valid URL.")

# Footer
st.write("---")
st.markdown('<p class="footer">Developed with ❤️ for secure web experiences | © 2024</p>', unsafe_allow_html=True)
