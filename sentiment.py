import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Function to analyze sentiment and return sentiment & score
def analyze_sentiment(text):
    result = pipe(text)[0]
    return result['label'], result['score']

# Function for emoji based on sentiment
def get_emoji(sentiment):
    if sentiment == "POSITIVE":
        return "😊"
    elif sentiment == "NEGATIVE":
        return "😢"
    elif sentiment == "NEUTRAL":
        return "😐"
    else:
        return "🤔"

# Streamlit app title
st.title("Sentiment Analysis with Accurate Emoji")
st.write("---")

# Input section (text and CSV)
user_text = st.text_input("Enter text for sentiment analysis:")

# Text button placed directly after input box
if st.button("Analyze Text"):
    if user_text:
        sentiment, score = analyze_sentiment(user_text)
        emoji = get_emoji(sentiment)

        # Display sentiment result with emoji
        st.write(f"Sentiment (Text Input): **{sentiment.upper()}** {emoji}")
        st.write(f"Confidence Score: **{score:.2f}**")

# CSV file uploader and button
uploaded_file = st.file_uploader("CSV file analysis (Column name must be 'whatsapp.text.body'):")

# CSV button placed directly after file uploader
if st.button("Analyze CSV") and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'whatsapp.text.body' not in df.columns:
            raise ValueError("CSV file must contain a column named 'whatsapp.text.body'")

        # Drop rows with NaN values
        df.dropna(subset=['whatsapp.text.body'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Apply sentiment analysis to the CSV column
        df['sentiment'], df['score'] = zip(*df['whatsapp.text.body'].apply(analyze_sentiment))
        df['emoji'] = df['sentiment'].apply(get_emoji)

        st.write("Sentiment Analysis Results (from uploaded CSV):")
        st.dataframe(df[['whatsapp.text.body', 'sentiment', 'score', 'emoji']])

    except ValueError as e:
        st.error(e)
