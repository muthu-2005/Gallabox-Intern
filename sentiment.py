import streamlit as st
from textblob import TextBlob
import pandas as pd

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    score = abs(sentiment)
    if sentiment > 0:
        return "Positive", score
    elif sentiment < 0:
        return "Negative", score
    else:
        return "Neutral", score

st.title("Sentiment Analysis")
st.write("---")

user_text = st.text_input("Enter text for sentiment analysis:")
uploaded_file = st.file_uploader("CSV file analysis:")

if st.button("Analyze"):
    sentiment_scores = []

    if user_text:
        sentiment, score = analyze_sentiment(user_text)
        st.write(f"Sentiment (Text Input): **{sentiment.upper()}**")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'whatsapp.text.body' not in df.columns:
                raise ValueError("CSV file must contain a column named 'whatsapp.text.body'")
            
            # Drop rows with NaN values in the 'whatsapp.text.body' column
            df.dropna(subset=['whatsapp.text.body'], inplace=True)

            # Reset index without creating a new column for the old index
            df.reset_index(drop=True, inplace=True)

            df['sentiment'], df['score'] = zip(*df['whatsapp.text.body'].apply(analyze_sentiment))
            sentiment_scores.extend(df['score'])

            st.write("Sentiment Analysis Results (from uploaded CSV):")
            st.dataframe(df[['whatsapp.text.body', 'sentiment', 'score']])

        except ValueError as e:
            st.error(e)  

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        if avg_sentiment > 0.1:
            conclusion = "Positive"
        elif avg_sentiment < -0.1:
            conclusion = "Negative"
        else:
            conclusion = "Neutral"

        st.write("---")
        st.write("Conclusion: The conversation is predominantly", conclusion.upper())
