# coding: utf-8

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import itertools

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from collections import Counter
import re
import datetime as dt
from datetime import date
from datetime import datetime

import requests
from bs4 import BeautifulSoup

import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TweetTokenizer
from nltk import tokenize

from wordcloud import WordCloud
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import MiniBatchKMeans

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectPercentile

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc,precision_score, accuracy_score, recall_score, f1_score
from scipy import interp

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import push_notebook, show, output_notebook
import bokeh.plotting as bplt

import lda
import pyLDAvis
import pyLDAvis.gensim

import warnings
warnings.filterwarnings("ignore")
import logging

import gensim
from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec 
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models.ldamodel import LdaModel

from copy import deepcopy
from pprint import pprint

from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, LSTM, Activation
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence

import pickle
import os

print(os.getcwd())

# Importing tweets from csv into dataframe
# (wczytanie danych tweetow z csv do ramki)

try:
    tweets = pd.read_csv('bezrobocie_tweets_15.08.2017.csv', names = ['username', 'date', 'retweets', 'favorites', 'text', 'geo', 'mentions', 'hashtags', 'id', 'permalink'], sep=";", 
                         skiprows=1, encoding='utf-8')
except ValueError as exc:
    raise ParseError('CSV parse error - %s', parser_context=None)

print(tweets.head())
tweets.text[1]

# Removing duplicates from dataframe
# (usuniecie duplikatow tweetow z ramki)
#print('before', len(tweets)) # 21069 

tweets.drop_duplicates(['text'], inplace=True)
print(len(tweets)) # 20803 


# Separating the time variable by hour, day, month and year for further analysis using datetime 
# (podzial zmiennej data na godzine, rok, miesiac)

tweets['date'] = pd.to_datetime(tweets['date'])
tweets['hour'] = tweets['date'].apply(lambda x: x.hour)
tweets['month'] = tweets['date'].apply(lambda x: x.month)
tweets['day'] = tweets['date'].apply(lambda x: x.day)
tweets['year'] = tweets['date'].apply(lambda x: x.year)
tweets['length'] = tweets["text"].apply(len)
tweets['num_of_words'] = tweets["text"].str.split().apply(len)
# addding 1 column for counting
# (dodanie 1 kolumny do zliczania)
tweets['dummy_count'] = 1
tweets.head(5)


# Changing date into string 
# (zamiana daty na string)

tweets['time_decoded'] = pd.to_datetime(tweets.date)
tweets['time_decoded'] = tweets.time_decoded.map(lambda x: x.strftime('%Y-%m-%d'))
tweets[['date', 'time_decoded']].head()


# Who twitted most about the unemployment in Poland from 2015 
# (Kto najwiecej twittowal o bezrobociu od poczatku 2015 roku) 

grouped = pd.DataFrame(tweets.groupby('username').size().rename('counts')).sort_values('counts', ascending=False)
grouped.head()


# There are several users who twitted a lot about unemployment in Poland

# Who twitted most about the unemployment in Poland from 2015 - creating a plot
# (Kto najwiecej twittowal o bezrobociu od poczatku 2015 roku) 

get_ipython().magic('matplotlib inline')
tweets_by_username = tweets['username'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xlabel('Username', fontsize=12)
ax.set_ylabel('Number of tweets' , fontsize=12)
ax.set_title('Top 5 usernames twitting about unemployment', fontsize=12, fontweight='bold')
tweets_by_username[:8].plot(ax=ax, kind='bar', color='blue')


# During holidays people tweet more often

# Number of tweets by months - creating a plot
# (Liczba tweetow po miesiacach - wykres)

tweets['monthYear'] = tweets['date'].apply(lambda x: str(x.year)+'-'+str(x.month)+'-1')
tweets['monthYear'].unique()[0]
tweets['monthYear'] = pd.to_datetime(tweets['monthYear'])

tweets[['date', 'monthYear']].head()

yrMonthly_tweets2 = tweets.groupby(['month','year']).size().unstack()


plt.figure(figsize=(10,5))

plt.title('Number of tweets by months', fontsize=12, fontweight='bold')
plt.bar(yrMonthly_tweets2.index, yrMonthly_tweets2.values[:,0])
plt.bar(yrMonthly_tweets2.index+12, yrMonthly_tweets2.values[:,1], color='g')
plt.bar(yrMonthly_tweets2.index[:8]+24, yrMonthly_tweets2.values[:,2][~np.isnan(yrMonthly_tweets2.values[:,2])], color='r')
plt.xticks(list(yrMonthly_tweets2.index)+list(yrMonthly_tweets2.index+12)+list(yrMonthly_tweets2.index[:8]+24))
plt.show() 

x = yrMonthly_tweets2.values[:,1]
x = x[~np.isnan(x)]



# People in Poland tweet mostly on Tue, Wed, Thur about unemployment

# Number of tweets during a day
# (liczba tweetow dziennie)

tweets['weekDay']= tweets['date'].map(lambda x: date.isocalendar(x)[2])
daily_tweets = tweets.groupby(['weekDay']).size()#.unstack()

plt.figure(figsize=(8,4))

plt.title('Number of tweets by days', fontsize=12, fontweight='bold')
plt.bar(daily_tweets.index, daily_tweets.values)
plt.xticks(daily_tweets.index)
plt.show()


# Web scraping

# Web scraping of dates of publication of Statistical Bulletin from the Polish Statistical Office website (stat.gov.pl)
# (pobranie daty publikacji Biuletynu Statystycznego ze strony GUS)

df = pd.DataFrame(columns=['Name','Date'])

for j in range(1,500):

    url = "http://stat.gov.pl/aktualnosci/5-years,"+str(j)+",arch.html"

    f = requests.get(url)
    soup = BeautifulSoup(f.text, 'html.parser')
    name = soup.find(text=re.compile("Biuletyn Statystyczny")) # Searching for publication

    if name !=None:

        name.strip()
        
        date = name.parent.find_next("div", class_="date").text.strip()                                                                                       
 
        df1 = pd.DataFrame({'Name':name, 'Date':[date]})
        df = df.append(df1)

# Writing the table
df = df.reset_index(drop=True)
df


# Sometimes the number of tweets increases after official publication of unemployment rate in Poland, but it not the rule

# Dates of publication of unemployment rate from the Statististal Bulletin from the Polish Statistical Office website (GUS)
# and number of tweets at that time of publication
# (daty publikacji stopy bezrobocia ze strony GUS i liczba tweetow w tym czasie)

nr_of_tweets = tweets.groupby('time_decoded').size() 
plt.figure(figsize=(18,5))
plt.plot(pd.to_datetime(nr_of_tweets.index), nr_of_tweets) 
official_dates = list(df['Date']) 
official_dates_stamp = pd.to_datetime(official_dates) 

# Cutting off dates below 2014 year
official_dates_stamp = official_dates_stamp[official_dates_stamp > datetime.strptime('01.01.2015', '%d.%m.%Y')]

komunikat = np.ones(len(official_dates_stamp))
plt.scatter(official_dates_stamp,komunikat, marker='^', s=20, color='darkred')
plt.title('Number of tweets after unemployment rate publication')
plt.show()


# This function returns True if a word ("unemployment") is found in the text, otherwise it returns False 
# (funkcja, ktora sprawdzi, czy slowo "bezrobocie" jest w tekscie tweeta)
# In 20798 tweets there is word "bezrobocie"

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

# Next, we will add column to tweets DataFrame
tweets['bezrobocie'] = tweets['text'].apply(lambda tweet: word_in_text('bezrobocie', tweet))
print(tweets['bezrobocie'].value_counts()[True]) 


# Number of co-occurences of phrase "unemployment in Poland" (in Polish)"
# (liczba współwystępowań slow: bezrobocie w Polsce)
# "bezrobocie w Polsce" is in 1283 tweets

counter=0
for line in tweets.iloc[:,4]:
    position =re.findall(r'(bezro.+?) (w+?) (pols.+?)', line.lower(), re.UNICODE)
    if position !=[]:
        #print(pozycja)
        counter +=1
print(counter) 


print(range(len(tweets)))


# Preprocessing data using NLTK:
# Clean, Tokenize, Remove stopwords, Stem, Lemmatize tweets


# Function for "cleaning" tweets - text pre-processing
# (funkcja na "czyszczenie" tweetow)

def cleaning(s):
    s = str(s)
    s = re.sub(r'<.+?>',' ',s)  # usuniecie znacznikow html
    s = re.sub(r'\{[~\{^\{]+?\}',' ',s)  # usuniecie nawiasow klamrowych 
    s = re.sub(r'\n',' ', s) # usuniecie newlines
    s = re.sub(r'\\',' ', s) # usuniecie backslashes
    s = re.sub(r'[\,\.\"\-\']',' ',s) # usuniecie przecinkow, kropek i innych
    s = re.sub("\d+", "", s) # usuniecie cyferek
    s = re.sub('[!%@#$_]', '', s)
    s = re.sub(r'#\S+', ' ', s) # usuniecie hashtagow
    s = re.sub(r'\s{2,}',' ', s)
    return s


# Adding column with "cleaned" tweets
# (dodanie kolumny z "czystymi" tweetami)

tweets['cleaning'] = [cleaning(s) for s in tweets['text']]
tweets['cleaning'].head()
#type(tweets['cleaning'][0])


tweets.ix[:, ['username', 'text', 'cleaning', 'length', 'num_of_words']].head(10)


# Tokenizing tweets, removing Polish stopwords, punctuation
# (tokenizacja tweetow, usuniecie polskich "stopwords")

stopwords = pd.read_csv('Polish_Stopwords.txt')

def tokenizer(text):
    try:
        tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
        
        tokens = []
        
        for token_by_sent in tokens_:
            tokens += token_by_sent
        
        tokens = list(filter(lambda t: t.lower() not in stopwords.values, tokens))
        
        tokens = list(filter(lambda t: t not in punctuation, tokens))                     
        tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', 
                                            u'\u2014', u'\u2026', u'\u2013'], tokens))
        
        tokens = list(filter(lambda t: '/' not in t, tokens))      
        
        tokens = list(filter(lambda t: len(t)>2, tokens))

        filtered_tokens = []
        
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))

        return filtered_tokens
    
    except Error as e:
        print(e)


# A new column 'tokens' is created using the map method applied to the 'cleaning' column
# The tokenizer function has been applied to each "cleaned" text through all rows 
# Each resulting value is then put into the 'tokens' column that is created after the assignment 
# (dodanie nowej kolumny z tokenami na "oczyszczonych" tweetach)

tweets['tokens'] = tweets.cleaning.map(tokenizer)


tweets.ix[:, ['text', 'cleaning', 'tokens']].head(10)


#  First 5 descriptions of text and tokens look the following:
# (5 pierwszych tweetow i tokeny tekstow)

for text, tokens in zip(tweets['text'].head(5), tweets['tokens'].head(5)):
    print('text:', text)
    print('tokens:', tokens)
    print() 


# Stemming tweets and lemmatization
# (lematyzacja - sprowadzanie slow do rdzenia)
# source: https://github.com/MarcinKosinski/trigeR5/blob/master/dicts/polimorfologik-2.1.zip/polimorfologik-2.1.txt

words = pd.read_csv('polimorfologik-2.1.txt', sep=';', header=None)
words.columns
words = words.ix[:, [0,1]]
words = words.set_index([1])
dictionary = words.ix[:].to_dict(orient='dict')


# Function changing a word into its primary form 
# (zamiana slow na forme podstawowa)

def word_to_stem(word):
    if word in dictionary[0]:
        output = dictionary[0][word]
    else: output = word
    return output

word_to_stem('polskiego')


# Adding column with stemmed words by applying def word_to_stem on 'tokens' column
# (dodanie kolumny po lematyzacji)

tweets['stem'] = tweets['tokens'].map(lambda a: [word_to_stem(w) for w in a])

#print( tweets['tokens'][4])
#print( tweets['stem'][4])


tweets.ix[:, ['username', 'text', 'tokens', 'stem']].head(5)


# The most common tokens are: unemployment, work, falling, Poland, the lowest

# Finding most_common 100 words/tokens by applying a word count function
# (znalezienie 100 najczesciej wystepujacych slow/tokenow)

def keywords():
    tokens = tweets['tokens']
    tokens_all = []
    for each_token in tokens:
        tokens_all += each_token
    counter = Counter(tokens_all)
    return counter.most_common(10)

keywords()


# Generating Word Cloud - input

list1 = []

for a in tweets['stem']:
    list1 +=a

    list2 = ''
for a in list1:    
    list2 = list2+' '+a


# Generating Word Cloud 
# (generowanie chmury słow)

#from wordcloud import WordCloud
#from PIL import Image

wordcloud = WordCloud(max_font_size=40, relative_scaling=0.3).generate(list2[:100000])
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# Text processing - CountVectorizer

# Pickling results for machine learning - backup

X = [' '.join(a) for a in tweets['stem']] 

with open('articles.txt', 'wb') as ap:
    pickle.dump(X, ap)


# Converting a collection of stemmed tweets to a matrix of token counts
# (Konwersja kolekcji tweetów na macierz liczby tokenow)

X = [' '.join(a) for a in tweets['stem']] 

c = CountVectorizer(token_pattern='(?u)\\b\\w+\\b',min_df=3,max_df=0.5) 
# min_df=3 means "ignore terms that appear in less than 3 documents"
# max_df=0.5 means "ignore terms that appear in more than 50% of the documents"
dtm = c.fit(X)
art = dtm.transform(X) 


# Function for top ten words 
# (funkcja zliczaca 10 najczesciej wystepujacych słów)

def top_words(M,dtm,k):
    words = np.array(dtm.get_feature_names())
    return(np.array([words[np.squeeze(np.array(np.argsort(M[i,:].todense())))[-k:]] for i in range(M.shape[0])]))


top_words(art,dtm,10)


# Vocabulary (słownik)
vocab = c.get_feature_names()
np.array(vocab[:30])


# Number of index(row)
# (index tj. numer wiersza)
dtm.vocabulary_['bezrobo'] 


# Visualize the most frequent 10 words
words_cv = [(w, i, art.getcol(i).sum()) for w, i in dtm.vocabulary_.items()]
words_cv = sorted(words_cv, key=lambda x: -x[2])[0:99]
words_cv[:8]


# Text processing - TF IDF
# Tf-idf stands for term frequencey-inverse document frequency. It's a numerical statistic intended to reflect
# How important a word is to a document or a corpus (i.e a collection of documents)

X = [' '.join(a) for a in tweets['stem']] 

vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 1))
vz = vectorizer.fit_transform(X)
# min_df is minimum number of documents that contain a term t
# max_features is maximum number of unique tokens (across documents) that we'd consider
# TfidfVectorizer preprocesses texts using the tokenizer we defined above 

vz.shape 


# vz is a tfidf matrix:
# - its number of rows is the total number of documents (list of stemmed tweets)
# - its number of columns is the total number of unique terms (tokens) across the documents 

vz.shape[1] # number of columns 


# Vocabulary (słownik)
words = vectorizer.get_feature_names()
words[0:5]

#vectorizer.vocabulary_

# Dictionary mapping the tokens to their tfidf values
tfidfy = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidfy = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidfy), orient='index')
tfidfy.columns = ['tfidf']
tfidfy.head(5)


# Visualisation of distribution of the tfidf scores through a histogram
get_ipython().magic('matplotlib inline')
tfidfy.tfidf.hist(bins=50, figsize=(15,7))


# Getting the tf-idf values of features (tokens that have the lowest tfidf scores)
words_tfidf = [(w, i, vz.getcol(i).sum()) for w, i in vectorizer.vocabulary_.items()]
words_tfidf = sorted(words_tfidf, key=lambda x: -x[2])[0:99]
words_tfidf[:9]


# Tokens that have the lowest tfidf scores
# These are very common across many tweets
tfidfy.sort_values(by=['tfidf'], ascending=True).head(10)


# Tokens with highest tfidf scores
# Less common words. These words carry more meaning
tfidfy.sort_values(by=['tfidf'], ascending=False).head(10)


# SVD - dimension reduction (redukcja wymiarów)

# Tweets (documents) have more than 2000 features (see the vz shape), ie.each document has more than 2000 dimensions

# Singular Value Decomposition (SVD) is to reduce the dimension of each vector to 50 and then using t-SNE to reduce the dimension from 50 to 2
# (SVD - dekompozycja, rozkład macierzy)

svd = TruncatedSVD(n_components=50, random_state=0)
svd_tfidf = svd.fit_transform(vz)
svd_tfidf.shape

svd.explained_variance_ratio_
print(svd_tfidf.shape)
len(svd_tfidf[:,1])


# Reducing the dimension from 50 to 2 
#(redukcja wymiaru z 50 do 2)

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

tsne_tfidf.shape


# Each text is now modeled by a two dimensional vector
tsne_tfidf


# Plotting with "Bokeh - Python interactive visualization library
# By hovering on each tweets cluster, we can see groups of texts of similar keywords and thus referring to the same topic
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
output_notebook()

plot_tfidf = bp.figure(plot_width=900, plot_height=700, title="tf-idf clustering of tweets",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])

tfidf_df['text'] = tweets['text']

plot_tfidf.scatter(x='x', y='y', source=tfidf_df)

bplt.figure()
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"text": "@text"}

show(plot_tfidf, notebook_handle=True)
push_notebook()


# K-Means clustering (algorytm k-średnich | centroidów)

# K-means clustering is a method of vector quantization that is popular for cluster analysis in data mining 
# Aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster
# MiniBatchKMeans is an alternative implementation that does incremental updates of the centers positions using mini-batches 
# (Kmeans służy do podziału danych wejściowych na z góry założoną liczbę klas.) 
# (Jest to jeden z algorytmów stosowany w klasteryzacji (grupowaniu) i jest częścią uczenia nienadzorowanego w Machine Learning)

num_clusters = 5
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)


# Five first tweets and the associated cluster

for (i, desc) in enumerate(tweets.text):
    if(i < 5):
        print("Cluster " + str(kmeans_clusters[i]) + ": " + desc + 
              "(distance: " + str(kmeans_distances[i][kmeans_clusters[i]]) + ")")
        print('--------------------------')
        


# Top features (words) that describe each cluster:

# Clusters 0, 1 seem to be about low & falling unemployment in Poland

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(num_clusters):
    print("Cluster %d:" % i, end='')
    for j in sorted_centroids[i, :10]:
        print(' %s' % terms[j], end='')
    print()


# Let's visualize the tweets, according to their distance from each centroid in K clusters

# To do this, we need to reduce the dimensionality of kmeans_distances, using t-SNE again to reduce the dimensionality from 5 down to 2

# Reducing the dimensionality of kmeans_distances

tsne_kmeans = tsne_model.fit_transform(kmeans_distances)


# Colorizing each tweet according to the cluster it belongs to - using bokeh

colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
"#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of tweets",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
kmeans_df['cluster'] = kmeans_clusters
kmeans_df['text'] = tweets['text']

plot_kmeans.scatter(x='x', y='y', 
                    color=colormap[kmeans_clusters], 
                    source=kmeans_df)

bplt.figure()
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"text": "@text", "cluster":"@cluster"}
show(plot_kmeans,notebook_handle=True)
push_notebook()


# Clusters are separated
# By hovering on each one of them you can see the corresponding texts
# They deal appx. with the same topic
# There are some overlaps between different clusters

# Latent Dirichlet Allocation (LDA)

# Topic modeling algorithm called LDA to uncover the latent topics in tweets

# The number of topics needs to be specified upfront

#logging.getLogger("lda").setLevel(logging.WARNING)

cvectorizer = CountVectorizer(min_df=3, max_features=10000, tokenizer=tokenizer, ngram_range=(1,1))

cvz = cvectorizer.fit_transform(tweets['text'])

n_topics = 10
n_iter = 100
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
X_topics = lda_model.fit_transform(cvz)


# We can inspect the words that are most relevant to a topic

# Topics are dealing mostly with: low and/or falling unemployment in Poland (topic 2), people having jobs (topic 0,8) and growing economy (topic 4, 5)

# Topics in more detail below

n_top_words = 8
topic_summaries = []

topic_word = lda_model.topic_word_ 
vocab = cvectorizer.get_feature_names() 
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' '.join(topic_words))) 
    

lda_model.components_ 
lda_model.loglikelihood() 


# To visualize the tweets according to their topic distributions, we first need to reduce the dimensionality down to 2 using t-SNE

tsne_lda = tsne_model.fit_transform(X_topics)


# Let's get the main topic for each tweet
doc_topic = lda_model.doc_topic_
lda_keys = []
for i, tweet in enumerate(tweets['text']):
    lda_keys += [doc_topic[i].argmax()]


# Which we'll use to colorize them:
plot_lda = bp.figure(plot_width=700, plot_height=600, title="LDA topic visualization",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

lda_df = pd.DataFrame(tsne_lda, columns=['x','y'])
lda_df['text'] = tweets['text']

lda_df['topic'] = lda_keys
lda_df['topic'] = lda_df['topic'].map(int)

plot_lda.scatter(source=lda_df, x='x', y='y', color=colormap[lda_keys])
bplt.figure()
hover = plot_lda.select(dict(type=HoverTool))
hover.tooltips={"text":"@text", "topic":"@topic"}
show(plot_lda,notebook_handle=True)
push_notebook()

# Somewhat better separation between the topics
# No dominant topic

# Visualization of topics using pyLDAvis
# Visualization to explore LDA topics using pyldavis
    
lda_df['len_docs'] = tweets['tokens'].map(len)

def prepareLDAData():
    data = {
        'vocab': vocab,
        'doc_topic_dists': lda_model.doc_topic_,
        'doc_lengths': list(lda_df['len_docs']),
        'term_frequency':cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    } 
    return data


ldadata = prepareLDAData()

#import pyLDAvis
pyLDAvis.enable_notebook()

prepared_data = pyLDAvis.prepare(**ldadata)

pyLDAvis.save_html(prepared_data,'pyldadavis.html')

prepared_data


# Paragraph Vectors (doc2vec) - gensim
# "Paragraph Vector is an unsupervised algorithm that learns fixed-length feature representations from variable-length pieces of texts, such as sentences, paragraphs, and documents
# The algorithm represents each document by a dense vector which is trained to predict words in the document."

# Let's use doc2vec to created paragraph vectors for each tweet in the dataset
tknzr = TweetTokenizer()

# Let's extract the documents and tokenize them (without removing the stopwords):
docs = [TaggedDocument(tknzr.tokenize(cleaning(tweets['text'])), [i]) for i, tweet in enumerate(tweets)]

#docs[0] # here's what a tokenized document looks like

# We train our doc2vec model with 100 dimensions, a window of size 8, a minimum word count of 5 and with 4 workers
doc2vec_model = Doc2Vec(docs, size=100, window=8, min_count=5, workers=4)


# Similar words to" unemployment", "falling" are: still, next, year, will be

doc2vec_model.most_similar(positive=["bezrobocie", "spada"], negative=["a"])


# Let' see what each paragraph vector looks like:
doc2vec_model.docvecs[0]


# Now we're going to use t-SNE to reduce dimensionality and plot the tweets:
doc_vectors = [doc2vec_model.docvecs[i] for i, t in enumerate(tweets[:10000])]

tsne_d2v = tsne_model.fit_transform(doc_vectors)

# Let's put the tweets on a 2D plane:
            
plot_d2v = bp.figure(plot_width=900, plot_height=700, title="Tweets (doc2vec)",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

plot_d2v.scatter(x=tsne_d2v[:,0], y=tsne_d2v[:,1],
                    color=colormap[lda_keys][:10000],
                    source=bp.ColumnDataSource({
                        "tweet": tweets['text'],
                        "processed": tweets['cleaning']
                    })) # tweets['cleaning']

hover = plot_d2v.select(dict(type=HoverTool))
hover.tooltips={"tweet": "@tweet (processed: \"@processed\")"}
show(plot_d2v, notebook_handle=True)
push_notebook()


# LDA in gensim module

# Creating the term dictionary of our courpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(tweets['stem'])
#dictionary.save('dictionary.dict')

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
corpus = [dictionary.doc2bow(text) for text in tweets['stem']]
#print(len(corpus))
mm = corpora.MmCorpus.serialize('corpus.mm', corpus)

dictionary.keys()[:10]  # ID words

dictionary[0]   # words

x = dictionary.values()
x

corpus[0] # documents as a tuple (id word, occurences of this word)

ldamodel = LdaModel(corpus=corpus,id2word=dictionary,num_topics=10,alpha="auto")

ldamodel.save('lda.model')
# later on - load trained model from file
#ldamodel =  models.LdaModel.load('lda.model')

topics = []
for doc in corpus:
    topics.append(ldamodel[doc])


# The average document mentions 5.8 topics and 77 percent of them mention 9 or fewer

lens = np.array([len(t) for t in topics])
print(np.mean(lens))
# the average document mentions 5.8 topics and

print(np.mean(lens <= 9))
# 77 percent of them mention 9 or fewer

# We can also ask what the most talked about topic in is. First collect some statistics on topic usage
counts = np.zeros(100)
for doc_top in topics:
    for ti,_ in doc_top:
        counts[ti] +=1
words = ldamodel.show_topic(counts.argmax(), 64)
words[0]

#Alternatively, we can look at the least talked about topic:
#words = ldamodel.show_topic(counts.argmin(), 64)

# Visualization of words - Word Cloud
wordcloud = WordCloud(max_font_size=40, relative_scaling=0.3).generate(str(words))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

ldamodel.print_topics(3)
#print(ldamodel.print_topics(num_topics=2, num_words=4))

# Another way of printing topics

for i in range(0, ldamodel.num_topics-1):
    print(ldamodel.print_topic(i))

	
ldamodel.save('topic.model')
from gensim.models import LdaModel
loading = LdaModel.load('topic.model')

# print all topics
ldamodel.show_topics()

# show 1 topic
ldatopics = ldamodel.show_topics(formatted=False)[0]
ldatopics

ldamodel.get_topic_terms(topicid=0)

ldamodel.get_document_topics(corpus[7]) # text representation


for i in range(len(tweets['stem'])):

    print(ldamodel.get_document_topics(corpus[i], minimum_probability=0))


# Visualizing LDA model with pyLDAvis. The area of the circles represents the prevelance of the topic
# The length of the bars on the right represents the membership of a term in a particular topic

pyLDAvis.enable_notebook()
tweets = pyLDAvis.gensim.prepare(ldamodel,corpus, dictionary)

tweets

pyLDAvis.save_html(tweets,'LDA-gensim.html')


# Sentiment analysis

# Backup for sentiment analysis
list1 = []

for a in tweets['stem']:
    list1 +=a

    list2 = ''
for a in list1:    
    list2 = list2+' '+a


# Creating exemplary vocabularies for positive (pos) and negative (neg) terms in Polish

# choosing from list1

pos2 = ['spadać','niski','mały','najniższy','niższy','mniejszy','najmniejszy','maleć','szybko','program',
       'spaść', 'spadlo', 'zmniejszać', 'stabilizować', 'nizsze', 'zmaleć']
       
neg2 = ['bieda','biedny','ciężko','źle','zły','problem','katastrofa','zwolnić','zamykać','wyjechać',
       'wyjechalo', 'brakować', 'wyzysk', 'ruiny','ruina', 'stracony']


# Counting optimism score basing on the lists above

pos_count = 0; neg_count = 0
for a in list1[:]:
    if a in pos2:
        pos_count +=1
    elif a in neg2:
        neg_count +=1
    else:
        pass
print(pos_count, neg_count)
print('Optimism score: {:.2f}'.format((pos_count-neg_count)/(pos_count+neg_count)))


# Function for a sentiment of a tweet
# (sentiment pojedynczego tweeta)

def sentiment(s):
    pos_count = 0
    neg_count = 0
    for a in s:
    #for lst in my_list:
        if a in pos2:
            pos_count+=1
        elif a in neg2:
            neg_count += 1
        else:
            pass
    #score = (pos_count-neg_count)/(pos_count+neg_count)
    score = (pos_count-neg_count)/(pos_count+neg_count+0.0000001)
    return score


# The 'sentiment' function has been applied to each "stemmed" text through all rows 
# Each resulting value is then put into the 'sentiment' column that is created after the assignment 
# (dodanie nowej kolumny z sentymentem na "stemowanych" tweetach)

tweets['sentiment'] = tweets['stem'].apply(lambda a: sentiment(a))
tweets.ix[:, ['stem', 'sentiment']].head(10)


# Rounding
tweets['sentiment'] = tweets['sentiment'].apply(lambda x: round(float(x), 0))
tweets.ix[:, ['stem', 'sentiment']].head(10)


# Grouping tweets by months
# (tweety po miesiącach)
groupedMonths = tweets.groupby('monthYear')['stem'].sum()

groupedMonths = pd.DataFrame(groupedMonths)
#groupedMonths

# Grouping tweets by months (increasing positive sentiment)


groupedMonths['sentiment'] = groupedMonths['stem'].apply(lambda a: sentiment(a))
#groupedMonths[groupedMonths['sentiment']<=0.7]

gus = pd.read_csv('Gus_bezrobocie.csv')

gus = gus.tail(10)

# Load the file into data frame
gus['data'] = pd.to_datetime(gus['data'])

plt.figure(figsize=(10,4))

#plt.plot(gus.data, gus.stopa, color='r', lw=3) # not a good comparison after all 
plt.plot(groupedMonths.index, groupedMonths['sentiment'], color='g', lw=3)

plt.show()

# positive tweets
positive = tweets[tweets['sentiment']>0.0][['sentiment','text']]
positive.shape 

# neutral tweets
#neutral = tweets[tweets['sentiment']==0][['sentiment','text']]
#neutral.shape # (11942, 2)


# more negative tweets
negative = tweets[tweets['sentiment']<=0.0][['sentiment','text']]

negative.shape 


# HashingVectorizer

# Using HashingVectorizer
token = HashingVectorizer(ngram_range=(1, 1))
token

all_token = token.fit_transform(positive['text'].append(negative['text']))

all_token

positive['text'] = 'good'
negative['text'] = 'bad'

prep = positive.append(negative)
prep.shape


# Modelling (HashingVectorizer)
# Modelling

X_train, X_test, y_train, y_test = train_test_split(all_token, prep['text'], random_state = 777, test_size=.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Implemented model: SGDClassifier - 95% accuracy
model = OneVsRestClassifier(SGDClassifier())
model
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
test_prediction = model.predict(X_test)
test_prediction
y_test.head()


# Testing K-fold (HashingVectorizer)

# Cross validation: to ensure the models are unbiased after train-test splitting
# 10 fold testing (PassiveAggressiveClassifier)

model0= PassiveAggressiveClassifier(n_iter=1)

cv_result = cross_val_score(model0, all_token, prep['text'], cv=10)

print(cv_result)
print ('min  is %.4f'%cv_result.min())
print ('max  is %.4f'%cv_result.max())
print ('mean is %.4f'%cv_result.mean())
print ()


# Implemented model: LinearSVC
# Model: LinearSVC with over 97% accuracy on 25% validate dataset

model = LinearSVC(verbose = 1)

model.fit(X_train, y_train)
print ('accuracy')
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

# Modelling on full data - 99% accuracy
model_all = LinearSVC(verbose = 1)

model_all.fit(all_token, prep['text'])
print ('accuracy on train dataset')
model_all.score(all_token, prep['text'])

print(model_all.score(X_test, y_test))


# Saving model
from sklearn.externals import joblib

from sklearn.externals import joblib
joblib.dump(token, 'token.pkl', protocol=2)
joblib.dump(model_all, 'token_SVM.pkl', protocol=2) 

print(classification_report(y_test, model.predict(X_test), target_names=['Negative','Positive']))


# Plotting confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(confusion_matrix(y_test, model.predict(X_test)), 
                      classes=['Negative','Positive'],
                      title='Confusion matrix, without normalization')



#print(confusion_matrix(y_test, model.predict(X_test)))

# Function for predicting sentiment of a tweet
def pred_new(text): 
    
    text = token.transform([text]) # token is the trained tokenizer such as Hashvectorizer
    result = model_all.predict(text) # model is the trained model
    final = 'Negative' if result[0]=='bad' else 'Positive'
    return final

pred_new('bezrobocie spada')

pred_new('spada')


# Word2vec and Keras

# Let's make a copy of tweets data frame 
tweets2 = tweets.copy(deep=True)
tweets2.head(2)

# Remove all columns except for two columns

cols = [col for col in tweets2.columns if col in ['sentiment', 'stem']]
df2 = tweets2[cols]

def postprocess(data, n=100000):
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

df2 = postprocess(df2)

df2.head(5)

# Let's define a training set and a test set 

xw_train, xw_test, yw_train, yw_test = train_test_split(np.array(df2.stem), np.array(df2.sentiment), test_size=0.2)

# Before feeding lists of tokens into the word2vec model, we must first turn them into LabeledSentence objects 

LabeledSentence = gensim.models.doc2vec.LabeledSentence

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

xw_train = labelizeTweets(xw_train, 'TRAIN')
xw_test = labelizeTweets(xw_test, 'TEST')


# Let's check the first element from x_train
# Each element is an object with two attributes: a list (of tokens) and a label

xw_train[0]


# Building the word2vec model from x_train i.e. the corpus
warnings.filterwarnings("ignore")
n_dim = 200
tweet_w2v = Word2Vec(size=n_dim, min_count=10) # model is initialized with the dimension of the vector space and min_count (a threshold for filtering words that appear less)
tweet_w2v.build_vocab([x.words for x in xw_train]) # vocabulary is created
tweet_w2v.train([x.words for x in xw_train],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter) # model is trained i.e. its weights are updated

tweet_w2v.corpus_count

# Saving the model

tweet_w2v.save('w2v.model')
# tweet_w2v = Word2Vec.load('w2v.model') # you can continue training with the loaded model


# Once the model is built and trained on the corpus of tweets, we can use it to convert words to vectors
# The word vectors are stored in a KeyedVectors instance in model.wv

tweet_w2v.wv['bezrobocie']  # numpy vector of a word


# Word2Vec provides a method named most_similar - given a word, this method returns the top n similar ones
tweet_w2v.most_similar('bezrobocie')

# Performing various word tasks with the model 

x5 = sorted(tweet_w2v.wv.most_similar(positive=['bezrobocie', 'spadać'], negative=['rosnąć']))
x5

x6 = sorted(tweet_w2v.wv.most_similar_cosmul(positive=['spadać'], negative=['rosnąć'])) 
x6

tweet_w2v.wv.doesnt_match("bezrobocie spadać rosnąć rafalska".split())

tweet_w2v.wv.similarity('bezrobocie', 'spadać')


# Once the model is built and trained on the corpus of tweets, we can use it to convert words to vectors. Example:
# tweet_w2v['dobry']

vocab = list(tweet_w2v.wv.vocab.keys())
np.array(vocab[:10])


# Bokeh for vizualization of word vectors
# When clicking on a point, you can see the corresponding word


# Defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

# Getting a list of word vectors
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

# Dimensionality reduction. Converting the vectors to 2d vectors

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# Putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]

# Plotting: the corresponding word appears when you hover on the data point
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf, notebook_handle=True)
push_notebook()

# Building a sentiment classifier

# Compute a weighted average where each weight gives the importance of the word with respect to the corpus

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in xw_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocabulary size:', len(tfidf))


# Let's define a function that, given a list of tweet tokens, creates an averaged tweet vector

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not in the corpus
            continue
    if count != 0:
        vec /= count
    return vec


# Convert xw_train and xw_test into list of vectors using this function
# Scale each column to have zero mean and unit standard deviation

train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, xw_train)])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, xw_test)])
test_vecs_w2v = scale(test_vecs_w2v)


# Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs_w2v, yw_train)

print('Test Accuracy: %.2f' % lr.score(test_vecs_w2v, yw_test))


# Keras | neural network (sieci neuronowe)

# Feed these vectors into a neural network classifier
# basic layer architecture (1)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))  # 'sigmoid'
model.add(Dense(1, activation='sigmoid'))               # 'softmax'

model.compile(optimizer='rmsprop',                      # 'adam', 'sgd'
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stoping = EarlyStopping(patience=3, monitor = "val_loss")

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_vecs_w2v, yw_train, epochs=20, batch_size=32, verbose=1, 
          callbacks=None, validation_split=0.0, validation_data=None)

print(model.summary())


# Tweet sentiment classifier using word2vec and Keras - combination of these two tools resulted in a ~82% classification model accuracy

model.evaluate(test_vecs_w2v, yw_test, verbose=2)

print("Accuracy: %.2f%%" % (model.evaluate(test_vecs_w2v, yw_test, verbose=2)[1]*100))

model.predict(test_vecs_w2v, batch_size=32, verbose=0)


# basic layer architecture (2) - worser results
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

model = Sequential()

model.add(Dense(32, activation='sigmoid', input_dim=200))

model.add(Dense(1, activation='softmax'))

model.compile(optimizer='sgd', # 'adam', 'sgd', 'rmsprop'
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_vecs_w2v, yw_train, epochs=10, batch_size=32, verbose=1, 
          callbacks=None, validation_split=0.0, validation_data=None)

print(model.summary())

model.evaluate(test_vecs_w2v, yw_test, verbose=2)

print("Accuracy: %.2f%%" % (model.evaluate(test_vecs_w2v, yw_test, verbose=2)[1]*100))


# Training models/predicting tweet's positive or negative sentiment


# Trying to choose the best algorithm

# Making another copy of tweets data frame 
tweets3 = tweets.copy(deep=True)
tweets3.head(2)
tweets3.shape


# Binary classification of a tweet
tweets3['sentiment'] = tweets3['sentiment'].replace({-1.0: 0, 0.0: 0, 1.0: 1})
tweets3['sentiment'][:10] 


# Cross validation - to select the model

# Testing several algorithms using its default parameters 

# Applying 10-fold cross validation on the training set to select the best method

# Later on, using the grid search approach for parameters tuning 

# Building and evaluating models for each combination of algorithm parameters 

# Creating list for best algorithms using its default parameters 
ScoreSummaryByModel = list()


# Function for model evaluation - based on 'cleaned', but not tokenized tweets

def ModelEvaluation (model,comment):
    
    scoring = 'accuracy'
    
    pipeline = Pipeline([('vect', CountVectorizer())
                  , ('tfidf', TfidfTransformer())
                  , ('model', model)])
    
    scores = cross_val_score(pipeline, tweets3['cleaning'], tweets3['sentiment'], cv=10, scoring=scoring) #cv=kfold 
    mean = scores.mean()
    std = scores.std()
    #The mean score and the 95% confidence interval of the score estimate (accuracy)
    ScoreSummaryByModel.append([comment,mean, std, "%0.3f (+/- %0.3f)" % (mean, std * 2)])
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


ModelEvaluation (MultinomialNB(),'Naive Bayes classifier')

ModelEvaluation (BernoulliNB(binarize=0.0),'Bernoulli Naive Bayes')

ModelEvaluation (SVC(kernel='linear'),'SVC, linear kernel')

ModelEvaluation (LinearSVC(),'LinearSVC')

ModelEvaluation (SGDClassifier(),'SGD')


# Below is the summary. LinearSVC and SVC (linear kernel) with default parameters returned the highest accuracy

df_ScoreSummaryByModel=DataFrame(ScoreSummaryByModel,columns=['Method','Mean','Std','Accuracy'])
df_ScoreSummaryByModel.sort_values(['Mean'],ascending=False,inplace=True)
df_ScoreSummaryByModel


# GridSearchCV - parameters tuning for classifiers

# Using the grid search approach for parameters tuning 

# Building and evaluating models for each combination of algorithm parameters

ScoreSummaryByModelParams = list()


# Function for optimizing parameters - based on 'cleaned', but not tokenized tweets

def ModelParamsEvaluation (vectorizer,model,params,comment):
    best_params = []
    scoring = 'accuracy'# 'f1'
    
    pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', model),
    ])
    
    # Finding the best parameters
    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, verbose=1, scoring=scoring) #cv=5, refit=True)

    grid_search.fit(tweets3['cleaning'], tweets3['sentiment'])
    best_params.append(grid_search.best_params_)
    
    print("Best score: {0}".format(grid_search.best_score_))
    print("Best parameters set:")
    
    best_parameters = grid_search.best_estimator_.get_params()
    
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        ScoreSummaryByModelParams.append([comment,grid_search.best_score_,"\t%s: %r" % (param_name, best_parameters[param_name])])
    
    #pipeline.set_params(**best_parameters)


# Bernoulli Naive Bayes

p = {'vect__analyzer':('char', 'char_wb'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((2, 2), (3, 3)), 
    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(CountVectorizer(),BernoulliNB(),p,'Bernoulli Naive Bayes')


# 3 chars is almost a word. The score is lower then for word analyzer


# Bernoulli Naive Bayes, analyzer=word
p = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (3, 3), (5,5),(2,5)), 
    'clf__alpha': (1,0.1,0.01,0.001,0.0001,0)}

ModelParamsEvaluation(CountVectorizer(analyzer='word'),BernoulliNB(),p,'Bernoulli Naive Bayes, analyzer=word')

# Tweets are short messages, therefore unigrams make sense. Using unigrams seems to be the best approach


# LinearSVC
p = {'vect__analyzer':('char', 'char_wb'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((2, 2), (3, 3)),
    'clf__C': (1,0.1,0.01,0.001,0.0001)
    }
ModelParamsEvaluation(CountVectorizer(),LinearSVC(),p,'LinearSVC')


# LinearSVC, analyzer=word
p = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (3, 3), (5,5),(2,5)),
    'clf__C': (1,0.1,0.01,0.001,0.0001)
    }
ModelParamsEvaluation(CountVectorizer(analyzer='word'),LinearSVC(),p,'LinearSVC analyzer=word')


# SVC, linear kernel
p = {'vect__analyzer':('char', 'char_wb'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((2, 2), (3, 3)),
     'clf__C': (1,0.1,0.01,0.001,0.0001)}
ModelParamsEvaluation (CountVectorizer(),SVC(kernel='linear'),p,'SVC, linear kernel, char')


# SVC, linear kernel, analyzer=word
p = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (3, 3), (5,5),(2,5)),
    'clf__C': (1,0.1,0.01,0.001,0.0001)}
ModelParamsEvaluation (CountVectorizer(analyzer='word'),SVC(kernel='linear'),p,'SVC, linear kernel, analyzer=word')


# SGDClassifier
p = {'vect__analyzer':('char', 'char_wb'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((2, 2), (3, 3)),
    'clf__alpha': (0.01,0.001,0.0001,0.00001, 0.000001),
    'clf__penalty': ('l1','l2', 'elasticnet')}
ModelParamsEvaluation (CountVectorizer(),SGDClassifier(),p,'SGD Classifier, analyzer=char')


# SGDClassifier, analyzer='word'
p = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (3, 3), (5,5),(2,5)),
    'clf__alpha': (0.01,0.001,0.0001,0.00001, 0.000001),
    'clf__penalty': ('l1','l2', 'elasticnet')}
ModelParamsEvaluation (CountVectorizer(analyzer='word'),SGDClassifier(),p,'SGD Classifier, analyzer=word')


# Below is the summary: 

# The highest 97% accuracy returned:
# - SGDClassifier (alpha: 1e-05, char 1-ngram)
# - LinearSVC (C=1, analyzer=word, char 1-ngram)
# - SVC with linear kernel (C=1, analyzer=word, char 1-ngram)


# Long running time of models


df_ScoreSummaryByModelParams=DataFrame(ScoreSummaryByModelParams,columns=['Method','BestScore','BestParameter'])
df_ScoreSummaryByModelParams.sort_values(['BestScore'],ascending=False,inplace=True)
df_ScoreSummaryByModelParams


# Let's apply the discovered best approach to test data set 
# Using SGDClassifier as best model
# Apply some score metrics 


tweet_train, tweet_test, sentiment_train, sentiment_test = train_test_split(tweets3['cleaning'], tweets3['sentiment'], test_size=0.25, random_state=42)

print(tweet_train.shape, tweet_test.shape, sentiment_train.shape, sentiment_test.shape)


#The best result for SGD classifier, 1-1 n-grams 
sgd_pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='word',ngram_range=(1, 1), max_df=0.5)),
    ('tfidf', TfidfTransformer()), 
    ('classifier', SGDClassifier(alpha=1e-05, penalty="l1"))])

sgd_pipeline.fit(tweet_train, sentiment_train)

predictions = sgd_pipeline.predict(tweet_test)


# Function for metrics
def PredictionEvaluation(sentiment_test,sentiment_predictions):
    print ('Precision: %0.3f' % (precision_score(sentiment_test,sentiment_predictions)))
    print ('Accuracy: %0.3f' % (accuracy_score(sentiment_test,sentiment_predictions)))
    print ('Recall: %0.3f' % (recall_score(sentiment_test,sentiment_predictions)))
    print ('F1: %0.3f' % (f1_score(sentiment_test,sentiment_predictions)))
    print ('Confussion matrix:')
    print (confusion_matrix(sentiment_test,sentiment_predictions))
    print ('ROC-AUC: %0.3f' % (roc_auc_score(sentiment_test,sentiment_predictions)))


PredictionEvaluation(sentiment_test,predictions)


sgd_pipeline.predict(["bezrobocie spada"])[0]


plot_confusion_matrix(confusion_matrix(sentiment_test, sgd_pipeline.predict(tweet_test)), 
                      classes=['Negative','Positive'],
                      title='Confusion matrix, without normalization')


# Final tuning parameters by pipeline - SGD Classifier
# based on 'cleaned', but not tokenized tweets
X1 = tweets3['cleaning']  
y1 = tweets3['sentiment'] 

X1_train = X[:14000]
X1_validate = X[14000:18000]
X1_test = X[18000:]

y1_train = y[:14000]
y1_validate = y[14000:18000]
y1_test = y[18000:]

pipeline1 = Pipeline([
    ('vect', CountVectorizer(analyzer='word')), 
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

parameters1 = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l1', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

# Find the best parameters for both the feature extraction and the classifier
# GridSearchCV - parameters tuning for choosen classifier #(szukanie optymalnej kombinacji parametrów)
grid_search1 = GridSearchCV(pipeline1, parameters1, n_jobs=2, verbose=1, 
                           scoring='accuracy')

grid_search1.fit(X1_validate, y1_validate)

# The best parameters passed back to the pipeline
print ('The best model results: %0.3f' % grid_search1.best_score_)

best_parameters1 = grid_search1.best_estimator_.get_params()

for param_name in sorted(parameters1.keys()):
    print(param_name, best_parameters1[param_name])
    
pipeline1.set_params(**best_parameters1)

# Train model on train data X_train and Y_train
pipeline1.fit(X1_train, y1_train)

y1_pred = pipeline1.predict(X1_test) # Predicting labels on 10% test data (przewidujemy etykiety na 10% zbiorze testowym)

# Calculate precision, recall, F1-score 
print(metrics.classification_report(y1_test, y1_pred, digits=3))

#print(confusion_matrix(y1_test,pipeline1.predict(X1_test)))

#print ('ROC-AUC: %0.3f' % (roc_auc_score(y1_test,y1_pred)))


PredictionEvaluation(y1_test,y1_pred)

pipeline1.predict(["bezrobocie spada w Polsce"])[0]



# Final tuning parameters by pipeline - LinearSVC
# based on 'cleaned', but not tokenized tweets
X2 = tweets3['cleaning'] 
y2 = tweets3['sentiment']  

X2_train = X[:14000]
X2_validate = X[14000:18000] 
X2_test = X[18000:]

y2_train = y[:14000]
y2_validate = y[14000:18000]
y2_test = y[18000:]

pipeline2 = Pipeline([
       ('vect', TfidfVectorizer()),
       ('sel', SelectPercentile()),
       ('clf', LinearSVC())
])

parameters2 = {
   'vect__max_df': (0.25, 0.5),
   'vect__ngram_range': ((1, 1), (1, 2), (1,3)),
   'vect__use_idf': (True, False),
   'sel__percentile': (10,30,50,100),
   'clf__C': (0.01, 1, 10),
   'clf__class_weight': ('balanced',None),
}

# Find the best parameters for both the feature extraction and the classifier
# GridSearchCV - parameters tuning for choosen classifier #(szukanie optymalnej kombinacji parametrów)
grid_search2 = GridSearchCV(pipeline2, parameters2, n_jobs=2, verbose=1, 
                           scoring='accuracy')

# The best parameters passed back to the pipeline
grid_search2.fit(X2_validate, y2_validate)
print('The best model results: %0.3f' % grid_search2.best_score_)

best_parameters2 = grid_search2.best_estimator_.get_params()

for param_name in sorted(parameters2.keys()):
    print(param_name, best_parameters2[param_name])
    
pipeline2.set_params(**best_parameters2)

# Train model on train data X_train and Y_train
pipeline2.fit(X2_train, y2_train)

y2_pred = pipeline2.predict(X2_test) # Predicting labels on 10% test data (przewidujemy etykiety na 10% zbiorze testowym)

# Calculate precision, recall, F1-score 
print(metrics.classification_report(y2_test, y2_pred, digits=3))

#print(confusion_matrix(y2_test,pipeline2.predict(X2_test)))

#print ('ROC-AUC: %0.3f' % (roc_auc_score(y2_test,y2_pred)))


PredictionEvaluation(y2_test,y2_pred)



# Final tuning parameters by pipeline - SGD Classifier (only on train, test data set) + cross validation
# based on 'cleaned', but not tokenized tweets
msg_train, msg_test, label_train, label_test = train_test_split(tweets3['cleaning'], tweets3['sentiment'], 
                                                                  test_size=0.2, random_state=42)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

pipeline_sgd = Pipeline([
    ('vect', CountVectorizer(analyzer='word')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# pipeline parameters to automatically explore and tune
parameters_sgd = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l1', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

results_sgd = []

cv_results_sgd = cross_val_score(pipeline_sgd,  
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring or cv=kfold
                         scoring='accuracy',  # scoring metric
                         n_jobs=-1,  # -1 = use all cores = faster
                         )

results_sgd.append(cv_results_sgd)
#print(cv_results_sgd) 

msg_sgd = "%f (%f)" % (cv_results_sgd.mean(), cv_results_sgd.std())
#print(msg_sgd)

grid_sgd = GridSearchCV(
    pipeline_sgd,  # pipeline from above
    param_grid=parameters_sgd,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # scores we sre optimizing
    cv=StratifiedKFold(label_train, n_folds=5),  # type of cross validation to use
)

grid_sgd.fit(msg_train, label_train)    
print('The best model results: %0.3f' % grid_sgd.best_score_)

best_parameters_sgd = grid_sgd.best_estimator_.get_params()

for param_name in sorted(parameters_sgd.keys()):
    print(param_name, best_parameters_sgd[param_name])
    
pipeline_sgd.set_params(**best_parameters_sgd)

pipeline_sgd.fit(msg_train, label_train)  

y_pred_sgd = pipeline_sgd.predict(msg_test)     
print(metrics.classification_report(label_test, y_pred_sgd, digits=3)) 


PredictionEvaluation(label_test,y_pred_sgd)


grid_sgd.predict(["bezrobocie spada"])[0]

