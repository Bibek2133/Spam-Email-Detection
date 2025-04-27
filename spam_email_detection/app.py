import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📧",
    layout="centered",
)

# Custom CSS to center titles and change fonts
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    h1 {
        text-align: center;
        color: #2C3E50;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Title
st.title("📧 Spam Message Classifier")

st.markdown("""
Welcome to the **Spam Message Classifier**!  
Type or paste your message below, and we'll tell you if it's **Spam** or **Not Spam**.
""")

# User input
user_input = st.text_area("Enter your message:", height=150, key="user_message")

# Predict button
if st.button("Predict", key="predict_button"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message before clicking Predict.")
    else:
        with st.spinner('Analyzing message...'):
            # Preprocess and predict
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == 1:
            st.error("🚫 This is a **Spam** message!")
        else:
            st.success("✅ This is **Not a Spam** message.")

# Footer
st.markdown("---")
st.markdown("""
Made by Bibek Ranjan Sahoo
Powered by Streamlit 
""")
