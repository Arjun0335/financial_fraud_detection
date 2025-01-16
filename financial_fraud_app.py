import streamlit as st
import pickle  # Use this if the model is saved as a pickle file
from sklearn.feature_extraction.text import CountVectorizer

# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
    with open("C:/Users/arjun/Assignment/spam_text.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("C:/Users/arjun/Assignment/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Initialize the model and vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("Spam Message Detector")
st.write("Enter a text message below to check if it's spam or not.")

# Text Input
message = st.text_area("Enter your message here:")

# Prediction Logic
if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Preprocess and predict
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        import re
        port_stem = PorterStemmer()
        def stemming (content):
            stemmed_content = re.sub('[^a-zA-z]',' ', content)
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
            stemmed_content = ' '.join(stemmed_content)
            return stemmed_content
        message = stemming(message)
        transformed_message = vectorizer.transform([message])  # Transform the input text
        prediction = model.predict(transformed_message)
        
        # Display Result
        if prediction[0] == 'spam':
            st.error("🚨 This message is likely SPAM!")
        else:
            st.success("✅ This message is NOT spam.")

# Additional Options
st.sidebar.title("About")
st.sidebar.info("This tool uses a machine learning model to classify text messages as SPAM or NOT SPAM.")
