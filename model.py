import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("sample30.csv")
tfidf_vectorizer=pd.read_pickle('tfidf_vectorizer.pkl')
user_final_rating_df=pd.read_pickle('user_final_rating.pkl')
product_mapping_df=pd.read_pickle('product_mapping.pkl')
model=pd.read_pickle('model_lr.pkl')   

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]','',text)
    text = re.sub(r'[^\w\s]', '',text)
    text = re.sub(r'[\d]','',text)
    text = re.sub(r'[*x]?','',text)
    text = re.sub('[0-9]+', '', text)    
    return text

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):    
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def preprocess_and_lemmatize(input_text):
    input_text = basic_cleaning(input_text)
    words = nltk.word_tokenize(input_text)
    words = remove_stopwords(words)
    lemmas = lemmatize_verbs(words)
    return ' '.join(lemmas)

def top_20_user_recommendations(user_input): 
    # Calculate top 20 recommendations
    d = user_final_rating_df.loc[user_input].sort_values(ascending=False)[0:20]
    d = pd.merge(d, product_mapping_df, left_on='id', right_on='id', how='left')   
    data_selected = data[data.id.isin(d.id.tolist())]
    output_df = data_selected[['id', 'name', 'reviews_text']]
    output_df['reviews_text'] = output_df['reviews_text'].astype(str)
    output_df['reviews_text_lemmatized'] = output_df['reviews_text'].apply(preprocess_and_lemmatize)
    output_df['predicted_sentiment'] = output_df['reviews_text_lemmatized'].apply(lambda x: predict_complaint(x)[0])
     # From the top 20, Calculate top 5 recommendations based on sentiment analysis    
    top_5_recommendations = calculate_top_5_recommendations(output_df)
    return top_5_recommendations

def predict_complaint(sentence):
    word_vect_custom = tfidf_vectorizer.transform(pd.Series(sentence))
    word_vect_custom_df = pd.DataFrame(word_vect_custom.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
    custom_pred = model.predict(word_vect_custom_df)      
    # Check if predictions are not empty
    if len(custom_pred) > 0:
        return custom_pred
    else:    
        return [-1]

def calculate_top_5_recommendations(top_20_df):
    temp=top_20_df.groupby('id').sum()    
    temp['positive_percent']=temp.apply(lambda x: x['predicted_sentiment']/sum(temp['predicted_sentiment']), axis=1)    
    final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index    
    top_5 = data[data.id.isin(final_list)][['id', 'name','brand']].drop_duplicates()
    return top_5

# print(top_20_user_recommendations("elena79"))
