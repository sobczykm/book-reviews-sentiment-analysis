import streamlit as st
import pandas as pd
import plotly.express as px
import random
import csv
from model_inference import SentimentModel

page = st.sidebar.radio("", ["Home", "Score Match", "Book Finder"])


def load_random_sample_with_books(ratings_file="data/ratings.csv", books_file="data/books.csv", sample_size=1000):
    reservoir = []
    with open(ratings_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < sample_size:
                reservoir.append(row)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = row

    df_sample = pd.DataFrame(reservoir)

    df_sample['review/score'] = df_sample['review/score'].astype(float)
    if 'review/time' in df_sample.columns:
        df_sample['review/time'] = pd.to_datetime(df_sample['review/time'], unit='s', errors='coerce')

    df_books = pd.read_csv(books_file)

    if 'publishedDate' in df_books.columns:
        df_books['publishedDate'] = pd.to_datetime(df_books['publishedDate'], errors='coerce')

    df_merged = pd.merge(df_sample, df_books, on='Title', how='left')

    return df_merged

if "df_sample" not in st.session_state:
    with st.spinner("Generating the sample please hold"):
        st.session_state.df_sample = load_random_sample_with_books()
df_sample = st.session_state.df_sample

if page == "Home":
    st.title("Amazon Reviews")

    # --- Podstawowe statystyki ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of books", df_sample["Title"].nunique())
    col2.metric("Averge Score", round(df_sample["review/score"].mean(), 2))
    col3.metric("Averge number of reviews", round(df_sample["ratingsCount"].mean(), 2))


    fig_scores = px.histogram(
        df_sample,
        x="review/score",
        nbins=5,  # 5 binów = oceny 1-5
        labels={"review/score": "Ocena", "count": "Liczba książek"},
        title="Rozkład ocen książek"
    )
    fig_scores.update_traces(marker_line_width=0.5)  
    fig_scores.update_xaxes(dtick=1) 
    fig_scores.update_layout(bargap=0.3)  

    st.plotly_chart(
        fig_scores,
        use_container_width=True,
        config={
            'displayModeBar': False
        }
    )

    st.subheader("Rozkład liczby recenzji")
    ratings_count = df_sample.groupby("Title")["ratingsCount"].sum().reset_index()

    # Kategorie ręczne
    bins = [0, 10, 20, 50, 100, float('inf')]
    labels = ["0-10", "10-20", "20-50", "50-100", "100+"]

    ratings_count['ratings_bin'] = pd.cut(
        ratings_count['ratingsCount'], 
        bins=bins, 
        labels=labels, 
        right=False
    )

    fig_ratings = px.histogram(
        ratings_count,
        x="ratings_bin",
        category_orders={"ratings_bin": labels},
        labels={"ratings_bin": "Liczba recenzji", "count": "Liczba książek"},
        title="Ile książek ma daną liczbę recenzji"
    )
    fig_ratings.update_traces(marker_line_width=0.5)  
    fig_ratings.update_layout(bargap=0.3)

    st.plotly_chart(
        fig_ratings,
        use_container_width=True,
        config={
            'displayModeBar': False
        }
    )

# --- Score Match ---
elif page == "Score Match":
    st.title("Score Match")
    text = st.text_area("Write your review:")
    
    # Load model in session state (cached to avoid reloading)
    if "sentiment_model" not in st.session_state:
        with st.spinner("Loading sentiment model..."):
            try:
                st.session_state.sentiment_model = SentimentModel()
            except FileNotFoundError as e:
                st.error(f"Model not found. Please run `train_model.py` first to train the model.")
                st.stop()
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()
    
    if st.button("Help me with the score"):
        if text.strip() == "":
            st.warning("Please write your review first")
        else:
            try:
                with st.spinner("Analyzing sentiment..."):
                    score, probs = st.session_state.sentiment_model.predict_sentiment(text, return_probs=True)
                    stars = "★" * score + "☆" * (5 - score)
                    st.markdown(f"## Proposed score: {stars} ({score}/5)")
                    
                    # Show probability distribution for debugging
                    with st.expander("View prediction probabilities"):
                        prob_dict = {f"Score {i+1}": f"{probs[i]:.2%}" for i in range(5)}
                        st.write(prob_dict)
            except Exception as e:
                st.error(f"Error predicting sentiment: {str(e)}")

elif page == "Book Finder":
    st.title("Book Finder")

    def clean_list(value):
        if pd.isna(value):
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v).strip()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return ""
            if value.startswith("[") and value.endswith("]"):
                return value[1:-1].replace("'", "").replace('"', "").strip()
            return value
        return str(value).strip() if value else ""

    if "df_books" not in st.session_state:
        with st.spinner("Please wait..."):
            st.session_state.df_books = df_sample.copy()

    df_merged = st.session_state.df_books

    # Zakres lat wydania
    if 'publishedDate' in df_merged.columns and pd.api.types.is_datetime64_any_dtype(df_merged['publishedDate']):
        min_year = int(df_merged['publishedDate'].dt.year.min())
        max_year = int(df_merged['publishedDate'].dt.year.max())
        year_range = st.slider(
            "Please select the time period:",
            min_year,
            max_year,
            (min_year, max_year)
        )
    else:
        year_range = (0, 9999)

    # Zakres średniej oceny
    min_rating, max_rating = st.slider(
        "Plese select the average score:",
        1.0,
        5.0,
        (1.0, 5.0),
        0.1
    )

    # Kategorie
    if 'categories' in df_merged.columns:
        cleaned_cats = df_merged['categories'].apply(clean_list)
        unique_categories = cleaned_cats[(cleaned_cats != "") & (cleaned_cats.notna())].unique()
        categories_options = ["Any"] + sorted([cat for cat in unique_categories if cat])
        selected_category = st.selectbox("Choose category:", categories_options)
    else:
        selected_category = "Any"

    # Popularność
    if 'ratingsCount' in df_merged.columns:
        min_ratings = int(df_merged['ratingsCount'].min())
        max_ratings = int(df_merged['ratingsCount'].max())
        ratings_range = st.slider(
            "Number of reviews:",
            min_ratings,
            max_ratings,
            (min_ratings, max_ratings)
        )
    else:
       ratings_range = (0, 0)

    # ---------- WYSZUKIWANIE ----------
    if st.button("Find me a book"):
        filtered = df_merged.copy()

        if 'publishedDate' in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered['publishedDate']):
            filtered = filtered[
                (filtered['publishedDate'].dt.year >= year_range[0]) &
                (filtered['publishedDate'].dt.year <= year_range[1])
            ]

        if 'review/score' in filtered.columns:
            filtered = filtered[
                (filtered['review/score'] >= min_rating) &
                (filtered['review/score'] <= max_rating)
            ]

        if selected_category != "Any" and 'categories' in filtered.columns:
            filtered = filtered[
                filtered['categories'].apply(clean_list) == selected_category
            ]

        if 'ratingsCount' in filtered.columns:
            filtered = filtered[
                (filtered['ratingsCount'] >= ratings_range[0]) &
                (filtered['ratingsCount'] <= ratings_range[1])
            ]

        # ---------- WYNIK ----------
        st.subheader(f"Number of books that match the criteria: {len(filtered)}")

        if len(filtered) > 0:
            random_book = filtered.sample(1).iloc[0]
            col1, col2 = st.columns([1, 3])

            with col1:
                img = random_book.get('image')
                if pd.notna(img):
                    st.image(img, width=450)
                else:
                    st.write("No cover found")

            with col2:
                st.markdown(f"### {random_book.get('Title', 'No Data')}")

                st.write(f"**AUthors:** {clean_list(random_book.get('authors', 'No Data'))}")
                st.write(f"**Genera:** {clean_list(random_book.get('categories', 'No Data'))}")

                description = random_book.get('description', 'No Data')
                st.write(f"**Synopsis:** {description}")

                score = random_book.get('review/score')
                st.write(f"**Average score:** {round(score, 2) if pd.notna(score) else 'No Data'}")

                ratings = random_book.get('ratingsCount')
                st.write(f"**Number of reviews:** {int(ratings) if pd.notna(ratings) else 'No Data'}")

                published = random_book.get('publishedDate')
                st.write(f"**Date:** {published.date() if pd.notna(published) else 'No Data'}")

                preview_link = random_book.get('previewLink')
                if pd.notna(preview_link):
                    st.markdown(f"[Link]({preview_link})")

        else:
            st.warning("No books match the criteria - please tweak your search or reload the app")
