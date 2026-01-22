import streamlit as st
import pandas as pd
import plotly.express as px
import random
import csv
from model_inference import SentimentModel

page = st.sidebar.radio("Navigation", ["Home", "Score Match", "Book Finder"])

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
    
    
    df_sample['review/time'] = pd.to_numeric(df_sample['review/time'], errors='coerce')
    df_sample['review/time'] = pd.to_datetime(df_sample['review/time'], unit='s', errors='coerce')

    df_books = pd.read_csv(books_file)

    df_books['publishedDate'] = pd.to_datetime(df_books['publishedDate'], errors='coerce')

    df_merged = pd.merge(df_sample, df_books, on='Title', how='left')

    return df_merged


if "df_sample" not in st.session_state: 
    with st.spinner("Generating the sample please hold"):
        st.session_state.df_sample = load_random_sample_with_books()
df_sample = st.session_state.df_sample

if page == "Home":
    st.title("Amazon Reviews")

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of books", df_sample["Title"].nunique())
    col2.metric("Averge Score", round(df_sample["review/score"].mean(), 2))
    col3.metric("Averge number of reviews", round(df_sample["ratingsCount"].mean(), 2))


    fig_scores = px.histogram(
        df_sample,
        x="review/score",
        nbins=5, 
        labels={"review/score": "Score"},
        title="Scores' distribution"
    )
    fig_scores.update_yaxes(title_text="Number of reviews")
    fig_scores.update_traces(marker_line_width=0.5)  
    fig_scores.update_xaxes(dtick=1) 
    fig_scores.update_layout(bargap=0.3)  

    st.plotly_chart(
        fig_scores,
        width="stretch",
        config={
            'displayModeBar': False
        }
    )

    ratings_count = df_sample.groupby("Title")["ratingsCount"].sum().reset_index() 

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
        labels={"ratings_bin": "Number of reviews"},
        title="Reviews per book"
    )
    fig_ratings.update_yaxes(title_text="Number of books")
    fig_ratings.update_traces(marker_line_width=0.5)  
    fig_ratings.update_layout(bargap=0.3)

    st.plotly_chart(
        fig_ratings,
        width="stretch",
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

    def normalize(value):
        if pd.isna(value):
            return "No Data"
        if isinstance(value, list):
            return ", ".join(value)
        return str(value).strip("[]").replace("'", "").replace('"', "")

    def apply_filters(
        df,
        year_range, min_year, max_year,
        min_rating, max_rating,
        ratings_range, min_r, max_r,
        selected_category
    ):
        filtered = df.copy()

        if year_range != (min_year, max_year):
            filtered = filtered[
                (filtered['publishedDate'].notna()) &
                (filtered['publishedDate'].dt.year >= year_range[0]) &
                (filtered['publishedDate'].dt.year <= year_range[1])
            ]

        if (min_rating, max_rating) != (1.0, 5.0):
            filtered = filtered[
                (filtered['review/score'].notna()) &
                (filtered['review/score'] >= min_rating) &
                (filtered['review/score'] <= max_rating)
            ]

        if selected_category != "Any":
            filtered = filtered[
                filtered['categories']
                .fillna("")
                .apply(normalize)
                .str.contains(selected_category, case=False, na=False)
            ]

        if ratings_range != (min_r, max_r):
            filtered = filtered[
                (filtered['ratingsCount'].notna()) &
                (filtered['ratingsCount'] >= ratings_range[0]) &
                (filtered['ratingsCount'] <= ratings_range[1])
            ]

        return filtered

    st.title("Book Finder")

    if "df_books" not in st.session_state:
        st.session_state.df_books = df_sample.copy()

    df = st.session_state.df_books

    min_year = int(df['publishedDate'].dt.year.min())
    max_year = int(df['publishedDate'].dt.year.max())
    year_range = st.slider("Publication year:", min_year, max_year, (min_year, max_year))

    min_rating, max_rating = st.slider("Average score:", 1.0, 5.0, (1.0, 5.0), 0.1)

    categories = ["Any"] + sorted(df['categories'].dropna().apply(normalize).unique())
    selected_category = st.selectbox("Category:", categories)

    min_r = int(df['ratingsCount'].min())
    max_r = int(df['ratingsCount'].max())
    ratings_range = st.slider("Number of reviews:", min_r, max_r, (min_r, max_r))

    max_books = st.number_input(
        "Max books to draw:",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    if st.button("Find books"):

        filtered = apply_filters(
            df=df,
            year_range=year_range,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            max_rating=max_rating,
            ratings_range=ratings_range,
            min_r=min_r,
            max_r=max_r,
            selected_category=selected_category
        )

        if len(filtered) == 0:
            st.session_state.filtered_books = None
        else:
            if len(filtered) > max_books:
                filtered = filtered.sample(max_books, replace=False)

            st.session_state.filtered_books = filtered.reset_index(drop=True)
            st.session_state.page = 1

    if "filtered_books" in st.session_state:

        if st.session_state.filtered_books is None:
            st.warning("There were no books matching your criteria. Please tweak your search or reload the page.")

        else:
            filtered = st.session_state.filtered_books
            total_pages = len(filtered)

            col_prev, col_page, col_next = st.columns([1, 2, 1])

            with col_prev:
                if st.session_state.page > 1:
                    if st.button("Previous"):
                        st.session_state.page -= 1
                        st.rerun()

            with col_page:
                st.markdown(f"### Book {st.session_state.page} / {total_pages}")

            with col_next:
                if st.session_state.page < total_pages:
                    if st.button("Next"):
                        st.session_state.page += 1
                        st.rerun()

            book = filtered.iloc[st.session_state.page - 1]

            col1, col2 = st.columns([1, 2])

            with col1:
                if pd.notna(book.get("image")):
                    st.image(book["image"], width=200)
                else:
                    st.write("No cover")

            with col2:
                st.markdown(f"### {normalize(book.get('Title'))}")
                st.write(f"**Authors:** {normalize(book.get('authors'))}")
                st.write(f"**Genre:** {normalize(book.get('categories'))}")
                st.write(f"**Synopsis:** {normalize(book.get('description'))}")
                st.write(f"**Average score:** {round(book['review/score'],2) if pd.notna(book.get('review/score')) else 'No Data'}")
                st.write(f"**Number of reviews:** {int(book['ratingsCount']) if pd.notna(book.get('ratingsCount')) else 'No Data'}")
                st.write(f"**Date:** {book['publishedDate'].date() if pd.notna(book.get('publishedDate')) else 'No Data'}")
                if pd.notna(book.get("previewLink")):
                    st.markdown(f"[Preview link]({book['previewLink']})")


