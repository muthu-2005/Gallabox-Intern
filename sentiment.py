import streamlit as st
import pandas as pd
from transformers import pipeline
import time

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(text):
    result = pipe(text)[0]
    sentiment = result['label']
    score = result['score'] if sentiment != 'NEGATIVE' else -result['score']  # Negative sentiment handling
    return sentiment, score

# Streamlit app title
st.title("Sentiment Analysis")
st.write("---")

# Text Input Section
user_text = st.text_input("Enter text for sentiment analysis:")
if st.button("Analyze Text"):
    if user_text:
        sentiment, score = analyze_sentiment(user_text)
        st.write(f"Sentiment (Text Input): **{sentiment.upper()}**")
        st.write(f"Sentiment Score: **{score:.2f}**")

        # Emoji rain effect based on sentiment
        if sentiment == "POSITIVE":
            st.balloons()
        elif sentiment == "NEGATIVE":
            for _ in range(3):
                st.error("💔")
                time.sleep(0.5)
        else:
            st.info("😐")

# CSV File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file for analysis:")

if st.button("Analyze CSV"):
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            
            # Ensure the column name is 'whatsapp.text.body'
            if 'whatsapp.text.body' not in df.columns:
                raise ValueError("CSV file must contain a column named 'whatsapp.text.body'")

            # Drop rows with NaN values in the 'whatsapp.text.body' column
            df.dropna(subset=['whatsapp.text.body'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Analyze sentiment for each text in the CSV
            df['sentiment'], df['score'] = zip(*df['whatsapp.text.body'].apply(analyze_sentiment))
            
            # Display results in the Streamlit app
            st.write("Sentiment Analysis Results (from uploaded CSV):")
            st.dataframe(df[['whatsapp.text.body', 'sentiment', 'score']])

            # Calculate the average sentiment score
            avg_sentiment = df['score'].mean()
            
            # Conclusion based on the average sentiment score
            if avg_sentiment > 0.1:
                conclusion = "Positive"
                st.balloons()  # Emoji effect for positive sentiment
            elif avg_sentiment < -0.1:
                conclusion = "Negative"
                for _ in range(3):
                    st.error("💔")  # Emoji effect for negative sentiment
                    time.sleep(0.5)
            else:
                conclusion = "Neutral"
                st.info("😐")  # Neutral sentiment effect

            st.write("---")
            st.write(f"Conclusion: The conversation is predominantly **{conclusion.upper()}**")

        except ValueError as e:
            st.error(e)
    else:
        st.error("Please upload a valid CSV file.")

# Note:
# - The "whatsapp.text.body" column must be present in the uploaded CSV file for analysis.
# - Emoji effects are triggered based on the sentiment scores.
