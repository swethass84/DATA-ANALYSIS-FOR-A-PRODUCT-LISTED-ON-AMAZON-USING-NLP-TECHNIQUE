# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 00:19:46 2022

@author: chandrika
"""

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from pickle import dump,load
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="NLP Sentiment Analysis", layout="wide")

def fetch_data():
    link=st.text_input("Enter product review link")
    url=link
    if not url:
        st.stop()
    if url:   
        r = requests.get(url)
        # Parsing the HTML content
        soup = BeautifulSoup(r.text, 'html.parser')

        # Getting desired data from our parsed soup
        reviews = soup.find_all('div', {'data-hook': 'review'})
        
        # Initialize list
        reviewlist = []
        
        def get_soup(url):
            r = requests.get(url,params={'url': url, 'wait': 2})
            soup = BeautifulSoup(r.text, 'html.parser')
            return soup
                
            df = pd.DataFrame(reviewlist)
        
        # Initialize list to store reviews data later on
        reviewlist = []
            
        # Function 2: look for web-tags in our soup, then append our data to reviewList
        def get_reviews(soup):
            reviews = soup.find_all('div', {'data-hook': 'review'})
            try:
                for item in reviews:
                    review = {
                        'product': soup.title.text.replace('Amazon.ca:Customer reviews: ', '').strip(), 
                        'date': item.find('span', {'data-hook': 'review-date'}).text.strip(),
                        'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
                        'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
                        'review_list': item.find('span', {'data-hook': 'review-body'}).text.strip(),
                    }
                    reviewlist.append(review)
            except:
                pass
            
        for x in range(0,100):
            soup = get_soup(url)
            get_reviews(soup)
            if not soup.find('li', {'class': 'a-disabled a-last'}):
                pass
            else:
                break

        df = pd.DataFrame(reviewlist)        
        return df 

def read_data():
    uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'], accept_multiple_files=False)
    
    if not uploaded_file:
        st.stop()
    if uploaded_file:
        file_container = st.expander("Check your uploaded .csv file")
        df = pd.read_csv(uploaded_file)
        file_container.write(df)
        #df = df.drop({'product'}, axis = 1)
        n=df.shape[0]
        st.write('Total number of reviews %5.00f' %(n))
    
        col1, col2 = st.columns(2)
        with col1:
            import seaborn as sns
            def countPlot():
                fig = plt.figure(figsize=(3,3.5))
                sns.countplot(data = df, x = 'rating')
                st.pyplot(fig)
            countPlot()

        with col2:
            def piePlot():
                df1 = pd.DataFrame(df['rating'].value_counts().reset_index())
                fig1 = plt.figure(figsize=(10, 4))
                plt.pie(data=df1,x='rating',labels='index', autopct='%.1f%%')  
                st.pyplot(fig1)
            piePlot()
    
    
        # Rating vs time
        date_ = []
        for i in range(n):
            date_.append(df.date[i][21:])
            df1 = pd.DataFrame(date_)
        df2 = df1.rename(columns = {0:'date_column'})
    
        New_columns = df2.join(df2['date_column'].str.split(' ', 2, expand=True).rename(columns={0:'Day', 1:'Month',2:'Year'}))
        combine = pd.concat([df, New_columns], axis=1, join='inner')
        combine['new_date_column'] = pd.to_datetime(combine['date_column'])
        new_date = pd.DataFrame(combine['new_date_column'])
        combine_1 = pd.concat([combine, new_date], axis=1)
        Avg_rating_month = combine_1.groupby('Month',as_index=False).agg({'rating':'mean'}) 
    
        # Bar graph
        import plotly.express as px
        #st.bar_chart(data=result)
        fig = px.bar(Avg_rating_month,x ='Month' ,y ='rating' ,title = "Mean Monthly Rating")
        st.plotly_chart(fig)
    
        review_text = []
        for i in range(n):
            review_text.append(df.review_list[i])
    
        #Lower Text
        clean_text_1 = []

        def to_lower_case(data):
            for words in review_text:
                clean_text_1.append(str.lower(words))
            to_lower_case(review_text)
    
        #Tokenisation
        from nltk.tokenize import sent_tokenize,word_tokenize
        import nltk
        nltk.download('punkt')
    
        sent_tok = []
        for sent in clean_text_1:
            sent = sent_tokenize(sent)
            sent_tok.append(sent)
        clean_text_2 = [word_tokenize(i) for i in clean_text_1]

        # Punctuation removal
        import re
        clean_text_3 =[]

        for words in clean_text_2:
            clean=[]
            for w in words :
                res = re.sub(r'[^\w\s]' ,"",w)
                if res !="":
                    clean.append(res)
            clean_text_3.append(clean)

        # Stopwords removal
        nltk.download('stopwords')
        from nltk.corpus import stopwords
    
        clean_text_4=[]

        for words in clean_text_3:
            w=[]
            for word in words:
                if not word in stopwords.words('english'):
                    w.append(word)
            clean_text_4.append(w)
        
        #Vader Sentiment Analyser    
        import re
        import os
        import sys
        import ast
    
        #plt.style.use('fivethirtyeight')
        #cp = sns.color_palette()
    
        # Function for getting the sentiment
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    
        emptyline=[]
        for row in df['review_list']: 
            vs=analyzer.polarity_scores(row)
            emptyline.append(vs)
        
        # Creating new dataframe with sentiments
        df_sentiments=pd.DataFrame(emptyline)
        df_sentiments.head()
        
        df_c = pd.concat([df.reset_index(drop=True), df_sentiments], axis=1)
        df_c['Sentiment'] = np.where(df_c['compound'] >= 0 , 'Positive','Negative')
        df_c = df_c.drop({'product','date','title'}, axis = 1)
        
        container = st.expander("Sentiment scores")
        container.write(df_c)
        # Bar graph
        import plotly.express as px
        result=df_c['Sentiment'].value_counts()
        fig = px.bar(result,x =result.index ,y =df_c['Sentiment'].value_counts(),color=result.index)
        st.plotly_chart(fig)




def main():
    
    image="https://th.bing.com/th/id/R.a72a0d3d582edcb7d61938ccffa44f75?rik=uzeXJRTMoZmOzQ&riu=http%3a%2f%2fama360.com%2fwp-content%2fuploads%2f2018%2f02%2famazon2.0.gif&ehk=ENOzqt9k1%2faordLd3kj2mpKpIm7vhSSCThvNP6t2SW4%3d&risl=&pid=ImgRaw&r=0"
    st.sidebar.image(image)
    menu=["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
        
    if choice=="Home":
        st.title("AMAZON PRODUCT SENTIMENT ANALYSIS")
        
        import webbrowser
        amazon = "https://www.amazon.in/"

        if st.button('Open browser'):
            webbrowser.open_new_tab(amazon)  
        df1 = fetch_data()
        #st.download_button(label="Download data as CSV", data=csv, file_name='Amazon_Reviews.csv',mime='text/csv')
        #if st.button('Download file as csv'):
        df1.to_csv(r'Amazon_reviews.csv', index=False)
        read_data()
    
    if choice=="About":
        st.header("Project 119 (Group 2)")
        st.subheader("Daily Data Analysis for a Product listed on Amazon")
        st.subheader('Mentor')
        st.write("_Rajshekar_")
        st.subheader("Group members")
        st.write("_Chandrika KJ_")
        st.write("_Deepak Kumar Sharma_")
        st.write("_H.Vishal_")
        st.write("_H.Shiva Nandini_")
        st.write("_Mohammed Mujtaba Hussain_")
        st.write("_Nilakantha Panigrahi_")
        st.write("_Swetha MV_")     
    
            
     
        
     
if __name__ == '__main__':
	main()        