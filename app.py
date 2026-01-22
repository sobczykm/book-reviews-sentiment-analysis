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


if "df" not in st.session_state: 
    with st.spinner("Generating the sample please hold"):
        st.session_state.df= load_random_sample_with_books()
df= st.session_state.df

if page == "Home":
    st.title("Amazon Reviews")

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of books", df["Title"].nunique())
    col2.metric("Averge Score", round(df["review/score"].mean(), 2))
    col3.metric("Averge number of reviews", round(df["ratingsCount"].mean(), 2))


    fig_scores = px.histogram(
        df,
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

    ratings_count = df.groupby("Title")["ratingsCount"].sum().reset_index() 

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

elif page == "Score Match":
    st.title("Score Match")
    text = st.text_area("Write your review:")
    
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
                    
                    with st.expander("View prediction probabilities"):
                        prob_dict = {f"Score {i+1}": f"{probs[i]:.2%}" for i in range(5)}
                        st.write(prob_dict)
            except Exception as e:
                st.error(f"Error predicting sentiment: {str(e)}")

elif page == "Book Finder":

    st.title("Book Finder")

    def normalize(value):
        if pd.isna(value):
            return "No Data"
        if isinstance(value, list):
            return ", ".join(value)
        return str(value).strip("[]").replace("'", "").replace('"', "")

    def apply_filters(year_range, rating_range, ratings_range,
                      min_year, max_year, min_r, max_r,
                      selected_category):

        filtered = df

        if year_range != (min_year, max_year):
            filtered = filtered[
                filtered['publishedDate'].notna() &
                (filtered['publishedDate'].dt.year >= year_range[0]) &
                (filtered['publishedDate'].dt.year <= year_range[1])
            ]

        
        if rating_range != (1.0, 5.0):
            filtered = filtered[
                filtered['review/score'].notna() &
                (filtered['review/score'] >= rating_range[0]) &
                (filtered['review/score'] <= rating_range[1])
            ]

        if ratings_range != (min_r, max_r):
            filtered = filtered[
                filtered['ratingsCount'].notna() &
                (filtered['ratingsCount'] >= ratings_range[0]) &
                (filtered['ratingsCount'] <= ratings_range[1])
            ]


        if selected_category != "Any":
            filtered = filtered[
                filtered['categories']
                .fillna("")
                .apply(normalize)
                .str.contains(selected_category, case=False, regex= False)
            ]

        return filtered


    min_year = int(df['publishedDate'].dt.year.min())
    max_year = int(df['publishedDate'].dt.year.max())
    year_range = st.slider("Publication year:", min_year, max_year, (min_year, max_year))

    rating_range = st.slider("Average score:", 1.0, 5.0, (1.0, 5.0), 0.1)

    categories = ["Any"] + sorted(df['categories'].dropna().apply(normalize).unique())
    selected_category = st.selectbox("Category:", categories)

    min_r = int(df['ratingsCount'].min())
    max_r = int(df['ratingsCount'].max())
    ratings_range = st.slider("Number of reviews:", min_r, max_r, (min_r, max_r))

    max_books = st.number_input("Max books to draw:", 1, 10, 5)

    if st.button("Find books"):

        result = apply_filters(
            year_range,
            rating_range,
            ratings_range,
            min_year,
            max_year,
            min_r,
            max_r,
            selected_category
        )

        if len(result) == 0:
            st.session_state.filtered_books = None
        else:
            if len(result) > max_books:
                result = result.sample(max_books)

            st.session_state.filtered_books = result
            st.session_state.page = 1


    if "filtered_books" in st.session_state:

        if st.session_state.filtered_books is None:
            st.warning("There were no books matching your criteria.")

        else:
            filtered = st.session_state.filtered_books
            total_pages = len(filtered)

            col_prev, col_page, col_next = st.columns([1, 2, 1])

            with col_prev:
                if st.session_state.page > 1 and st.button("Previous"):
                    st.session_state.page -= 1
                    st.rerun()

            with col_page:
                st.markdown(f"### Book {st.session_state.page} / {total_pages}")

            with col_next:
                if st.session_state.page < total_pages and st.button("Next"):
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
                st.write(f"**Category:** {normalize(book.get('categories'))}")
                st.write(f"**Synopsis:** {normalize(book.get('description'))}")
                st.write(f"**Average score:** {round(book['review/score'],2) if pd.notna(book.get('review/score')) else 'No Data'}")
                st.write(f"**Number of reviews:** {int(book['ratingsCount']) if pd.notna(book.get('ratingsCount')) else 'No Data'}")
                st.write(f"**Date:** {book['publishedDate'].date() if pd.notna(book.get('publishedDate')) else 'No Data'}")
                if pd.notna(book.get("previewLink")):
                    st.markdown(f"[Preview link]({book['previewLink']})")

