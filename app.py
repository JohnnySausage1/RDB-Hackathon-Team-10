import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load + Clean Data
# -----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # Clean dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop duplicates & nulls
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    df.dropna(how="all", inplace=True)
    after = df.shape[0]

    # Track cleaning stats
    cleaning_report = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "columns": list(df.columns)
    }

    return df, cleaning_report


# -----------------------------
# Challenge Functions
# -----------------------------
def challenge1_cleaning(report):
    st.subheader("🧹 Challenge 1: Data Cleaning")
    st.write("Rows before cleaning:", report["rows_before"])
    st.write("Rows after cleaning:", report["rows_after"])
    st.write("Removed:", report["rows_removed"])
    st.write("Columns available:", report["columns"])


def challenge2_exploration(df):
    st.subheader("🔍 Challenge 2: Data Exploration")
    st.write("Shape:", df.shape)
    st.write(df.head())


def challenge3_sentiment(df):
    st.subheader("😊 Challenge 3: Sentiment Classification")
    for col in ["Reviews", "Headline", "Opening Text"]:
        if col in df.columns:
            df["Sentiment"] = df[col].astype(str).apply(
                lambda x: "positive" if TextBlob(x).sentiment.polarity > 0
                else "negative" if TextBlob(x).sentiment.polarity < 0
                else "neutral"
            )
            st.bar_chart(df["Sentiment"].value_counts())
            break
    return df


def challenge4_keywords(df):
    st.subheader("🔑 Challenge 4: Keyword Analysis")
    keywords = []
    for col in ["Key Phrases", "Keywords"]:
        if col in df.columns:
            keywords.extend(df[col].dropna().astype(str).str.split(",").sum())
    keywords = [k.strip().lower() for k in keywords if k.strip()]
    if keywords:
        top_keywords = pd.Series(keywords).value_counts().head(10)
        st.bar_chart(top_keywords)


def challenge5_wordcloud(df):
    st.subheader("☁ Challenge 5: Word Clouds")
    text_data = ""
    for col in ["Key Phrases", "Keywords"]:
        if col in df.columns:
            text_data += " ".join(df[col].dropna().astype(str)) + " "
    if text_data.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        st.image(wc.to_array())


def challenge6_insights(df):
    st.subheader("📊 Challenge 6: Business Insights")
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    if "Source" in df.columns:
        top_sources = df["Source"].value_counts().head(1)
        col1.metric("🏆 Top Source", top_sources.index[0], int(top_sources.values[0]))
    if "Country" in df.columns and "Sentiment" in df.columns:
        positive_country = df[df["Sentiment"] == "positive"]["Country"].value_counts().head(1)
        if not positive_country.empty:
            col2.metric("🌍 Most Positive Country", positive_country.index[0], int(positive_country.values[0]))
    if "Keywords" in df.columns:
        most_common_keyword = df["Keywords"].dropna().astype(str).str.split(",").explode().str.strip().value_counts().head(1)
        if not most_common_keyword.empty:
            col3.metric("🔑 Top Keyword", most_common_keyword.index[0], int(most_common_keyword.values[0]))
    if "Date" in df.columns:
        busiest_month = df["Date"].dt.to_period("M").value_counts().head(1)
        if not busiest_month.empty:
            col4.metric("📅 Busiest Month", str(busiest_month.index[0]), int(busiest_month.values[0]))
    if "Source" in df.columns and "Reach" in df.columns:
        highest_reach = df.groupby("Source")["Reach"].mean().sort_values(ascending=False).head(1)
        if not highest_reach.empty:
            col5.metric("📡 Highest Avg Reach", highest_reach.index[0], round(highest_reach.values[0], 2))


def challenge7_predictive(df):
    st.subheader("🤖 Challenge 7: Predictive Sentiment (Simple Model)")
    for col in ["Reviews", "Headline", "Opening Text"]:
        if col in df.columns:
            X = df[col].astype(str)
            y = df["Sentiment"]
            vec = TfidfVectorizer(stop_words="english", max_features=1000)
            X_tfidf = vec.fit_transform(X)
            model = LogisticRegression(max_iter=500)
            model.fit(X_tfidf, y)
            user_text = st.text_input("Try a headline:")
            if user_text:
                pred = model.predict(vec.transform([user_text]))[0]
                st.success(f"Predicted Sentiment: **{pred}**")
            break


def playground_tab(df):
    st.subheader("🎉 Clean Data Playground")
    st.dataframe(df)
    st.download_button("Download Cleaned Data (CSV)", df.to_csv(index=False), "cleaned_dataset.csv", "text/csv")


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("📊 Hackathon Storyboard: Challenges 1–7")

    file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])
    if file:
        df, report = load_data(file)

        # Tabs per Challenge
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🧹 C1: Cleaning", "🔍 C2: Exploration", "😊 C3: Sentiment",
            "🔑 C4: Keywords", "☁ C5: Word Clouds", "📊 C6: Insights",
            "🤖 C7: Predictive", "🎉 Playground"
        ])

        with tab1: challenge1_cleaning(report)
        with tab2: challenge2_exploration(df)
        with tab3: df = challenge3_sentiment(df)
        with tab4: challenge4_keywords(df)
        with tab5: challenge5_wordcloud(df)
        with tab6: challenge6_insights(df)
        with tab7: challenge7_predictive(df)
        with tab8: playground_tab(df)


if __name__ == "__main__":
    main()
