# !/usr/bin/python3

import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tweepy  # https://github.com/tweepy/tweepy
import csv
from tkinter import *

fields = 'Tweets Name', 'No of Twees'

# Twitter API credentials
access_token = "966184838725283841-rGcLFB44BtP9SxKO3wmFyhGanGJd18F"
access_token_secret = "ivCrcr6Ntv5s9YzIPdKW6RMW6UYfLdsgFz61GXkLzkhnd"
consumer_key = "wnVx5DOTVL1VhAZ5gaoyyZodZ"
consumer_secret = "1ERcU51nvc1UpbmcwDaQJLgbubGBoIwTUqdDsYVTS4izaeJK92"

class Sentoments:

    def get_all_tweets(self,name, no_of_tweets):

        # Twitter only allows access to a users most recent 3240 tweets with 
        # this method
        # authorize twitter, initialize tweepy

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)

        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name=name, count=no_of_tweets)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            print ("getting tweets before %s" % (oldest))

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name=name, count=no_of_tweets, max_id=oldest)

            # save most recent tweets
            alltweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1

            print ("...%s tweets downloaded so far" % (len(alltweets)))

        # transform the tweepy tweets into a 2D array that will populate the csv
        # outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
        outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]
        # outtweets = [[tweet.text.replace("@", "")] for tweet in alltweets]

        # outtweets = outtweets.replace("@", "")
        # write the csv
        with open('%s_tweets.csv' % name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["tweet"])
            writer.writerows(outtweets)
        f.close

        pass

    # for read csv file

    def readfile(self, name):

        
        train = pd.read_csv('%s_tweets.csv' % name)
        # def word_count(self):
        # Number of Words
        train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
        word_count = train[['tweet', 'word_count']].head(n=30)
        print(word_count)

        # Number of characters
        train['char_count'] = train['tweet'].str.len()  # this also includes spaces
        charectars = train[['tweet', 'char_count']].head(n=30)
        print(charectars)
        # return (word_count,charectars)

        # Average Word Length
        def avg_word(sentence):
            words = sentence.split()
            return (sum(len(word) for word in words)/len(words))

        # def average(self):
        train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
        average_word = train[['tweet','avg_word']].head(n=30)
        print(average_word)

        # Number of stopwords;;;
        stop = stopwords.words('english')

        train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
        stop_word = train[['tweet','stopwords']].head(n=30)
        print(stop_word)

        # Number of special characters
        train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
        hash_tags = train[['tweet','hastags']].head(n=30)
        print(hash_tags)

        # Number of numerics
        train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
        numeric_word = train[['tweet','numerics']].head(n=30)
        print(numeric_word)

        # Number of Uppercase words
        train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
        upper_word = train[['tweet','upper']].head(n=30)
        print(upper_word) 

        # Pre-processing
        # Lower case
        train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
        train['tweet'].head(n=30)

        # Removing Punctuation
        train['tweet'] = train['tweet'].str.replace('[^\w\s]', '')
        train['tweet'].head(n=30)

        # Removal of Stop Words
        stop = stopwords.words('english')
        train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        train['tweet'].head(n=30)

        # Common word removal
        freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]

        freq = list(freq.index)
        train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        train['tweet'].head(n=30)

        # Rare words removal
        freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
        freq = list(freq.index)
        train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        train['tweet'].head(n=30)

        # Spelling correction
        train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

        # Tokenization
        words = TextBlob(train['tweet'][1]).words
        print(words)

        # Stemming
        st = PorterStemmer()
        train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

        # Lemmatization
        train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        train['tweet'].head(n=30)

        # N-grams
        TextBlob(train['tweet'][0]).ngrams(2)

        # Term frequency
        # TF = (Number of times term T appears in the particular row) / (number of terms in that row)
        tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
        tf1.columns = ['words', 'tf']
        print(tf1)

        # Inverse Document Frequency
        # IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.
        for i, word in enumerate(tf1['words']):
            tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))

        print(tf1)

        # Term Frequency – Inverse Document Frequency (TF-IDF)
        tf1['tfidf'] = tf1['tf'] * tf1['idf']
        print (tf1)


        tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words = 'english', ngram_range=(1,  1))
       
        train_vect = tfidf.fit_transform(train['tweet'])

        print(train_vect)

        # Bag of Words
        bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1, 1), analyzer = "word")
        train_bow = bow.fit_transform(train['tweet'])
        print(train_bow)

        # Sentiment Analysis
        # def sentiment_anaysis():
        train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
        train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0])

        sentiments = train[['tweet', 'sentiment']].head(n=30)
        print('{:*^20s}\n'.format('****Sentiment of Tweets****'), sentiments)
         # write the csv
        with open('sentiments.csv' , 'a') as f1:
            sentiments.to_csv(f1, header=False)
        pass
        print("****** Click to Quit for Exit ******")

    
def fetch(entries):
    list1 = []
    for entry in entries:
       
        text = entry[1].get()
        list1.append(text)
    obj = Sentoments()
    obj.get_all_tweets(list1[0], list1[1])
    obj.readfile(list1[0])

    # return list1

def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=15, text=field, anchor='w')
      ent = Entry(row)

      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
   return entries    

if __name__ == '__main__':

    root = Tk()
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
    root.title("Roumers Analysis")
    b1 = Button(root, text='Show', fg="green", command=(lambda e=ents: fetch(e)))
    b1.pack(side=RIGHT, padx=5, pady=5)
    b2 = Button(root, text='Quit', fg="red", command=root.quit)
    b2.pack(side=LEFT, padx=5, pady=5)
    root.mainloop()
