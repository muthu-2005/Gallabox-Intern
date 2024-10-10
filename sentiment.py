import streamlit as st
import pandas as pd
from transformers import pipeline
import time

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

# Function to simulate emoji rain effect (from top of the page)
def emoji_rain(emoji):
    for _ in range(10):  # Repeat the emoji rain effect 10 times for a longer rain
        st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
        time.sleep(0.3)  # Pause for a moment between each 'line' of emojis

# Streamlit app title
st.title("Sentiment Analysis with Emoji Rain")
st.write("---")

# Input section (text and CSV)
user_text = st.text_input("Enter text for sentiment analysis:")
uploaded_file = st.file_uploader("CSV file analysis (Column name must be 'whatsapp.text.body'):")

# Align buttons directly after input and file uploader
col1, col2 = st.columns([1, 1])
with col1:
    analyze_text = st.button("Analyze Text")
with col2:
    analyze_csv = st.button("Analyze CSV")

# Analyze text input
if analyze_text and user_text:
    sentiment, score = analyze_sentiment(user_text)
    emoji = get_emoji(sentiment)

    # Display sentiment result with emoji
    st.write(f"Sentiment (Text Input): **{sentiment.upper()}** {emoji}")
    st.write(f"Confidence Score: **{score:.2f}**")
    
    # Emoji rain effect based on sentiment
    emoji_rain(emoji)

# Analyze CSV input
if analyze_csv and uploaded_file is not None:
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

        # Show emoji rain for the first sentiment in CSV as an example
        first_sentiment_emoji = df['emoji'].iloc[0]
        emoji_rain(first_sentiment_emoji)

    except ValueError as e:
        st.error(e)
