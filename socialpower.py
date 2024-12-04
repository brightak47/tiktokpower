import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from wordcloud import WordCloud
import pydeck as pdk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Fetch Trending Data ---

# TikTok Trending
def fetch_trending_tiktok():
    url = "https://www.tiktok.com/trending"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    trending_data = []
    for video in soup.find_all("div", class_="video-feed-item-wrapper"):
        title = video.find("h3").text if video.find("h3") else "No title"
        video_url = video.find("a", href=True)["href"]
        trending_data.append({"title": title, "url": video_url})
    return trending_data

# Sentiment Analysis
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity, sentiment.subjectivity

# --- Streamlit App ---

st.title("Trending Content by Country")

# Sidebar Filters
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Select Country", ["US", "UK", "IN", "CA", "AU"])
search_query = st.sidebar.text_input("Search Hashtags or Topics", "")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
st.sidebar.checkbox("Enable Real-Time Updates", value=False)

# Theme Switcher
if theme == "Dark":
    st.markdown(
        """
        <style>
        .css-1d391kg { background-color: #333; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# TikTok Section
st.header("TikTok Trending")
tiktok_trends = fetch_trending_tiktok()
for trend in tiktok_trends:
    st.subheader(trend["title"])
    st.write(f"[Watch on TikTok]({trend['url']})")

# Search Feature
if search_query:
    st.header(f"Results for '{search_query}'")
    st.write("This feature is under development for TikTok and Instagram APIs.")

# Sentiment Analysis Section
st.header("Sentiment Analysis")
sample_text = "TikTok's latest trend is amazing!"
polarity, subjectivity = analyze_sentiment(sample_text)
st.write(f"Sentiment Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")

# Word Cloud
st.header("Trending Hashtag Cloud")
hashtags = " ".join(["#Dance", "#Music", "#Tech", "#Fashion", "#Style", "#Viral"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(hashtags)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Map Visualization
st.header("Trending Map")
map_data = pd.DataFrame({
    'Country': ['US', 'UK', 'IN'],
    'Lat': [37.0902, 55.3781, 20.5937],
    'Lon': [-95.7129, -3.4360, 78.9629],
    'Trend': ['#Dance', '#FashionWeek', '#Bollywood']
})
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=20,
        longitude=0,
        zoom=1,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[Lon, Lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=500000,
        ),
    ],
))

# Leaderboard
st.header("Trending Leaderboard")
leaderboard = pd.DataFrame({
    'Rank': [1, 2, 3],
    'Hashtag': ['#Dance', '#Fashion', '#Tech'],
    'Mentions': [12000, 9500, 8000]
})
st.table(leaderboard)

# ML Trend Prediction
st.header("Trend Predictions")
data = {'Day': [1, 2, 3, 4, 5], 'Mentions': [100, 150, 200, 300, 400]}
df = pd.DataFrame(data)
X = np.array(df['Day']).reshape(-1, 1)
y = df['Mentions']
model = LinearRegression().fit(X, y)
future_prediction = model.predict([[6]])[0]
st.write(f"Predicted Mentions for Day 6: {int(future_prediction)}")

# Notifications for New Trends
if st.button("Check for New Trends"):
    st.success("New trends are now available!")

# Share Button
st.header("Share Trends")
st.markdown('[Share on Twitter](https://twitter.com/intent/tweet?url=https://example.com&text=Check+this+out!)')
