# Sentiment-analysis-on-Twitter-data
Sentiment analysis on Twitter data

1. Purpose of the project

The aim of the project was to collect tweets about unemployment in Poland, predict their positive and negative sentiment and determine the best model for the tweets with unknown sentiment.

Data description:

The data comprises 21,000 tweets about unemployment collected for the period from 2015 till 15th August 2017 (in Polish).

2. Stages:

•	Reading the Twitter dataset from csv into a single data frame

•	Analysis – who twitted most about the unemployment in Poland from 2015, number of tweets by months

•	Web scraping of publication dates of the unemployment rate in Poland from the Polish Statistical Office website (www.stat.gov.pl)  -     included in the Statistical Bulletin 

•	Web scraping of the unemployment rate in Poland from the Polish Statistical Office website 
	(http://stat.gov.pl/obszary-tematyczne/rynek-pracy/bezrobocie-rejestrowane/stopa-bezrobocia-w-latach-1990-2017,4,1.html)  

•	Comparing those publication dates with the number of tweets at that time 

•	Processing tweets: cleaning, removing Polish stop words, punctuation, stemming 

•	Word Cloud of tweets

•	Machine learning / deep learning based on the tweets dataset:

- Feature extraction including the following:

	• Tokenization: breaking down the parsed tweet text into individual words / tokens

	• CountVectorizer: converting a collection of stemmed tweets to a matrix of token counts 

- TF-IDF: applying the TF-IDF algorithm to create feature vectors from the tokenized tweet texts 
	
- Performing dimension reduction to two dimensions using TruncatedSV, TSNE 
	
- Running the K-Means clustering algorithm – identifying tweet membership to clusters (on top of the tf-idf matrix) using the             MiniBatchKMeans algorithm and visualizing them with Bokeh
	
- Extracting topics in tweets using the Latent Dirichlet Allocation (LDA) algorithm and visualizing them using Bokeh and pyldavis
	
- LDA in gensim module
	
- Paragraph Vectors (doc2vec) implementation (gensim) in tweets
	
- Basic sentiment analysis of tweets: creating exemplary vocabularies for positive and negative terms in Polish, defining function for 	 a sentiment of a tweet resulting in labeling tweets as 1 when tweet was positive, 0 when negative 
	
- Word2Vec (group of Deep Learning models developed by Google with the aim of capturing the context of words) - applied to build a 		 sentiment classifier. The word2vec model learnt a representation for every word in this corpus, a representation that was later    	  on used to transform tweets into vectors
	
- Building neural network - using above representation of tweets to train a neural network classifier using Keras 
	
- Modelling - I tested several algorithms to predict sentiment of a tweet using cross validation with 10-folds and GridSearchCV   		(tuning parameters for chosen classifiers) and Pipeline. Calculated the accuracy, confusion matrix
	
3. Results:

•	SGDClassifier (analyzer=word, ngram (1,1), alpha: 1e-05), LinearSVC (C=1, analyzer=word, ngram (1,1), and SVC (linear kernel, C=1, 	   	 analyzer=word, ngram (1,1) ) returned approximately the same accuracy - 97-98%

•	Tweet sentiment classifier using word2vec and Keras - combination of these resulted in a ~82% classification model accuracy

•	Similarly, Bernoulli Naive Bayes - 82% accuracy

4. How to use

Run Python code "twitter_unemployment.py" or in jupyter notebook -> in code folder.

Data folder includes files that have to be imported into the "twitter_unemployment.py”, like: "bezrobocie_tweets_15.08.2017.csv" (21,000 tweets), "Polish_Stopwords.txt" and "Gus_bezrobocie.csv" (unemployment rate in Poland).

Additionally, bokeh plots have been attached: tfidf clustering of tweets, KMeans clustering of tweets, LDA topic visualization, map of 10,000 word vectors, Tweets (doc2vec), LDA-gensim, pyldadavis.

