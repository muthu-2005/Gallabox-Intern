import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Function to analyze sentiment and return sentiment & score
def analyze_sentiment(text):
    result = pipe(text)[0]
    return result['label'], result['score']

# Title and description
st.title("Sentiment Analysis with Emojis")
st.write("---")

# Emoji function based on sentiment
def get_emoji(sentiment):
    if sentiment == "POSITIVE":
        return "😊"
    elif sentiment == "NEGATIVE":
        return "😢"
    elif sentiment == "NEUTRAL":
        return "😐"
    else:
        return "🤔"

# Separate input options: text input and CSV upload
user_text = st.text_input("Enter text for sentiment analysis:")
uploaded_file = st.file_uploader("CSV file analysis (Column name must be 'whatsapp.text.body'):")

# Add buttons for separate actions
col1, col2 = st.columns(2)
with col1:
    analyze_text = st.button("Analyze Text")
with col2:
    analyze_csv = st.button("Analyze CSV")

# Analyze text input
if analyze_text and user_text:
    sentiment, score = analyze_sentiment(user_text)
    emoji = get_emoji(sentiment)
    st.write(f"Sentiment (Text Input): **{sentiment.upper()}** {emoji}")
    st.write(f"Confidence Score: **{score:.2f}**")

    # Emoji rain effect based on sentiment
    st.markdown(f"""
        <style>
        @keyframes emojiRain {{
            0% {{ transform: translateY(-100px); opacity: 1; }}
            100% {{ transform: translateY(100vh); opacity: 0; }}
        }}
        .emoji {{
            position: absolute;
            top: 0;
            font-size: 2rem;
            animation: emojiRain 4s linear infinite;
            animation-delay: calc(var(--i) * -0.5s);
        }}
        </style>
        <div style="position: relative; width: 100vw; height: 50vh; overflow: hidden;">
            <div class="emoji" style="--i: 1; left: 10%;">{emoji}</div>
            <div class="emoji" style="--i: 2; left: 30%;">{emoji}</div>
            <div class="emoji" style="--i: 3; left: 50%;">{emoji}</div>
            <div class="emoji" style="--i: 4; left: 70%;">{emoji}</div>
            <div class="emoji" style="--i: 5; left: 90%;">{emoji}</div>
        </div>
        """, unsafe_allow_html=True)

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

    except ValueError as e:
        st.error(e)

