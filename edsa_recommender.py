"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
#from array import arr

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from PIL import Image

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
train = pd.read_csv('resources/data/train.csv', nrows = 2000)
test = pd.read_csv('resources/data/test.csv', nrows = 2000)
gen_scores = pd.read_csv('resources/data/genome_scores.csv', nrows = 2000)
gen_tags = pd.read_csv('resources/data/genome_tags.csv', nrows = 2000)
imd = pd.read_csv('resources/data/imdb_data.csv', nrows = 2000)
links = pd.read_csv('resources/data/links.csv', nrows = 2000)
movies = pd.read_csv('resources/data/movies.csv', nrows = 2000)
tags = pd.read_csv('resources/data/tags.csv', nrows = 2000)
ratings_df = pd.read_csv('resources/data/ratings.csv', nrows = 2000)
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home Page","Introduction to EDA","EDA","Recommender System","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Recommender systems")
        image = Image.open('cp.png')
        st.image(image)
    # -------------------------------------------------------------------

    if page_selection == "Home Page":
        st.title("EDSA Movie Recommendation Wilderness")
        image = Image.open('moviess.jpg')
        st.image(image)
        st.header("Introduction")
        st.info("In today's technology-driven society, recommender systems are both socially and commercially important for ensuring that users can make informed decisions about the information they interact with on a regular basis. This is especially true in movie recommendations, where clever algorithms can help consumers locate exceptional films from tens of thousands of possibilities.")
        # You can read a markdown file from supporting resources folder
        st.info("With this background in mind, EDSA challenged  us to create a recommendation algorithm based on content or collaborative filtering that is capable of properly predicting how a user would score a movie they have yet to see based on their prior preferences.")

    # -------------------------------------------------------------------

    if page_selection == "Introduction to EDA":
        st.title("Exploratory Data Analysis (EDA)")
        image = Image.open('EDA.jpg')
        st.image(image)
        st.subheader("What is Exploratory Data Analysis?")
        st.markdown("Exploratory Data Analysis (EDA) is an approach/philosophy for data analysis that employs a variety of techniques (mostly graphical) to:")
        st.text("(I)    Maximize insight into a data set")
        st.text("(II)   Uncover underlying structure")
        st.text("(III)  Extract important variables")
        st.text("(IV)   Detect outliers and anomalies")
        st.text("(V)    Test underlying assumptions")
        st.text("(VI)   Develop parsimonious models")
        st.text("(VII)  Determine optimal factor settings")
        
    # -------------------------------------------------------------------

    if page_selection == "EDA":
        st.title("Exploratory Data Analysis")
        option = st.selectbox(
            'Which Raw Data would you like to Explore?',
            ('train', 'test', 'gen_scores','gen_tags','imd','links','movies','tags','ratings_df'))
        st.write('You selected:', option)

        if 'number_of_rows' not in st.session_state or 'type' not in st.session_state:
            st.session_state['number_of_rows'] = 5 
            st.session_state['type'] = 'Categorical'
            

        increament = st.button('Show more columns👆')
        if increament:
            st.session_state.number_of_rows += 10


        decrement = st.button('Show fewer columns👇')
        if decrement:
            st.session_state.number_of_rows -= 10

        if option == 'train':
            st.table(train.head(st.session_state['number_of_rows']))
        if option == 'test':
            st.table(test.head(st.session_state['number_of_rows']))
        if option == 'gen_scores':
            st.table(gen_scores.head(st.session_state['number_of_rows']))
        if option == 'gen_tags':
            st.table(gen_tags.head(st.session_state['number_of_rows']))
        if option == 'imd':
            st.table(imd.head(st.session_state['number_of_rows']))
        if option == 'links':
            st.table(links.head(st.session_state['number_of_rows']))
        if option == 'movies':
            st.table(movies.head(st.session_state['number_of_rows']))
        if option == 'tags':
            st.table(tags.head(st.session_state['number_of_rows']))
        if option == 'ratings_df':
            st.table(ratings_df.head(st.session_state['number_of_rows']))
        # ---------------------------------------------------------------
        #if st.button('Movie'):
            #wolf_mask = np.array(Image.open('wolf.jpg'))
            #dove_mask[230:250, 240:250]
            #fig1, ax1 = plt.subplots()
            #ax1.imshow(wolf_mask)
            #ax1.axis("off")
            #st.pyplot(fig1)

        # ---------------------- Wordclouds -----------------------------
        if st.button('Movie_titles'):
            st.write('##### WordCloud for Movie_titles')
            movies_word = movies['title'] = movies['title'].astype('str')
            #dove_mask = np.array(Image.open('wolf1.jpg'))
            movies_wordcloud = ' '.join(movies_word)
            title_wordcloud = WordCloud(stopwords = STOPWORDS,
                            background_color = 'White',
                            height = 1200,
                            width = 1900).generate(movies_wordcloud)
            # Display the generated image:
            fig1, ax1 = plt.subplots()
            ax1.imshow(title_wordcloud, interpolation='bilinear')
            ax1.axis("off")
            st.pyplot(fig1)

        if st.button('Movie_genres'):
            st.write('##### WordCloud for Movie_genres')
            movies['genres'] = movies['genres'].str.replace('|',' ')
            movies_word = movies['genres'] = movies['genres'].astype('str')
            #dove_mask = np.array(Image.open('wolf1.jpg'))
            movies_wordcloud = ' '.join(movies_word)
            title_wo = WordCloud(
                      colormap='winter',
                      background_color = 'White',
                      min_font_size=10).generate(movies_wordcloud)
            # Display the generated image:
            fig1, ax1 = plt.subplots()
            ax1.imshow(title_wo, interpolation='bilinear')
            ax1.axis("off")
            st.pyplot(fig1)
        if st.button('Top_20 actors'):
            st.title("Top 20 Actors Featured in the IMDB Database")
            image = Image.open('actors.png')
            st.image(image)
        if st.button('Movie production'):
            st.title("Number of movies produced annually")
            image = Image.open('movie_yr.png')
            st.image(image)
        if st.button('High rated movies'):
            st.title("Highly rated movies")
            image = Image.open('hi_movies.png')
            st.image(image)
        if st.button('Low rated movies'):
            st.title("Low rated movies")
            image = Image.open('low_movies.png')
            st.image(image)
        if st.button('Rating'):
            st.title("Most common ratings")
            image = Image.open('rating.png')
            st.image(image)
        if st.button('Users'):
            st.title("Grouped users")
            image = Image.open('users.png')
            st.image(image)
           

if __name__ == '__main__':
    main()
