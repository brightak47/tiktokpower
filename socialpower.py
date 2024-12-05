import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from wordcloud import WordCloud
import pydeck as pdk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from playwright.sync_api import sync_playwright
import time

# --- Fetch Trending Data ---

# TikTok Trending using Playwright Web Scraper
def fetch_trending_tiktok_via_playwright():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.tiktok.com/trending")
        time.sleep(5)  # Allow time for the page to load

        trending_data = []
        videos = page.query_selector_all("div.video-feed-item-wrapper a")
        for video in videos:
            title = video.get_attribute("title") or "No title"
            url = video.get_attribute("href")
            trending_data.append({"title": title, "url": url})

        browser.close()
    
    return trending_data

# Instagram Trending
def fetch_instagram_trending(access_token):
    instagram_business_id = "your-instagram-business-id"
    url = f"https://graph.facebook.com/v12.0/{instagram_business_id}/media?fields=id,caption,media_type,media_url,like_count,comments_count&access_token={access_token}"
    response = requests.get(url)
    data = response.json()

    trending_data = [
        {
            "caption": post.get("caption", "No caption"),
            "media_url": post.get("media_url"),
            "likes": post.get("like_count", 0),
            "comments": post.get("comments_count", 0),
        }
        for post in data.get("data", [])
    ]
    return trending_data

# Sentiment Analysis
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity, sentiment.subjectivity

# --- Streamlit App ---

st.set_page_config(page_title="Influencer Agent Dashboard", page_icon="📊", layout="wide")

st.title("Influencer Agent Dashboard")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("Filters")
st.sidebar.markdown("Select filters to refine the content.")
country = st.sidebar.text_input("Enter Country (e.g., US, UK, IN):", "US", help="Type the country code for trends.")
search_query = st.sidebar.text_input("Search Hashtags or Topics", "", help="Search specific hashtags or topics.")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"], help="Choose a theme for the app.")
access_token = st.sidebar.text_input("Instagram Access Token", type="password", help="Enter your Instagram Access Token.")
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
st.markdown(f"View the latest trending TikTok videos.")

tiktok_trends = fetch_trending_tiktok_via_playwright()
if tiktok_trends:
    for trend in tiktok_trends:
        st.subheader(trend["title"])
        st.write(f"[Watch on TikTok]({trend['url']})")
else:
    st.warning("No trending TikTok videos found. Please check your input or try again later.")

st.markdown("---")

# Instagram Section
st.header("Instagram Trending")
st.markdown("Explore the trending Instagram posts.")
if access_token:
    instagram_trends = fetch_instagram_trending(access_token)
    if instagram_trends:
        for trend in instagram_trends:
            st.subheader(trend["caption"])
            st.image(trend["media_url"], caption=f"Likes: {trend['likes']} | Comments: {trend['comments']}")
    else:
        st.info("No trending Instagram posts found.")
else:
    st.warning("Please provide an Instagram Access Token to fetch data.")

st.markdown("---")

# Sentiment Analysis Section
st.header("Sentiment Analysis")
st.markdown("Analyze the sentiment of trending content.")
sample_text = "TikTok's latest trend is amazing!"
polarity, subjectivity = analyze_sentiment(sample_text)
st.write(f"**Sentiment Polarity**: {polarity:.2f}, **Subjectivity**: {subjectivity:.2f}")

st.markdown("---")

# Word Cloud
st.header("Trending Hashtag Cloud")
st.markdown("Visualize popular hashtags as a word cloud.")
hashtags = " ".join(["#Dance", "#Music", "#Tech", "#Fashion", "#Style", "#Viral"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(hashtags)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

st.markdown("---")

# Map Visualization
st.header("Trending Map")
st.markdown("See trends geographically represented on the map.")
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

st.markdown("---")

# Leaderboard
st.header("Trending Leaderboard")
st.markdown("Discover the top hashtags ranked by mentions.")
leaderboard = pd.DataFrame({
    'Rank': [1, 2, 3],
    'Hashtag': ['#Dance', '#Fashion', '#Tech'],
    'Mentions': [12000, 9500, 8000]
})
st.table(leaderboard)

st.markdown("---")

# ML Trend Prediction
st.header("Trend Predictions")
st.markdown("Predict future popularity trends using machine learning.")
data = {'Day': [1, 2, 3, 4, 5], 'Mentions': [100, 150, 200, 300, 400]}
df = pd.DataFrame(data)
X = np.array(df['Day']).reshape(-1, 1)
y = df['Mentions']
model = LinearRegression().fit(X, y)
future_prediction = model.predict([[6]])[0]
st.write(f"**Predicted Mentions for Day 6**: {int(future_prediction)}")

st.markdown("---")

# Notifications for New Trends
if st.button("Check for New Trends"):
    st.success("New trends are now available!")

st.markdown("---")

# Share Button
st.header("Share Trends")
st.markdown("Spread the word about trending content.")
st.markdown('[Share on Twitter](https://twitter.com/intent/tweet?url=https://example.com&text=Check+this+out!)')
