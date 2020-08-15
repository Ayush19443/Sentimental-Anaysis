#%%
import sys
import tweepy
import csv
import numpy as np
import re
from textblob import TextBlob
import matplotlib.pyplot as plt, mpld3
from matplotlib import style
import matplotlib.animation as animation
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import PunktSentenceTokenizer 
from nltk.tokenize import PunktSentenceTokenizer 
import nltk 
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize 

# %%
class SA:
    def __init__(self):
        self.tweets = []
        self.tweetText = []
        
    
    def Data(self):
        consumer_key = '6ezxMc85nqfAleHzDwSH0jsFV'
        consumer_secret = 'Wg73disrWIn4DMEG5S35L1tJJSmrstC1f99HzmtXDccdJkAVgn'
        access_token = '870327487917633536-BPPxFDgURttwFtbS9wKMPZFoJ7TOZ1k'
        access_token_secret = 'pRGYIuXVdjsuHsghAAuzpWOMBJGRZ9b2PaB7BloQcOZaT'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)  
        auth.set_access_token(access_token, access_token_secret) 
        api = tweepy.API(auth) 

        word_search = input("Enter word to search: ")
        number_of_terms = int(input("Enter how many tweets to search: "))

        self.tweets = tweepy.Cursor(api.search, q=word_search, lang = "en").items(number_of_terms)

    
        
        #fo = open('result1.csv', 'a')
        #cw = csv.writer(fo)

        polarity=0
        positive=0
        negative=0
        neutral=0

        sid = SentimentIntensityAnalyzer()
        s=[]
        twe=[] 
        for i in self.tweets:
            self.tweetText.append(self.cleanTweet(i.text).encode('utf-8')) 
            sent_tokenizer = PunktSentenceTokenizer(i.text) 
            sents = sent_tokenizer.tokenize(i.text) 
            print(word_tokenize(i.text)) 
            print()

            print(sent_tokenize(i.text)) 
            print()
            print(i.text)
            
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
            scores = sid.polarity_scores(i.text)
            print(scores)
            analysis = TextBlob(i.text)
            print(analysis.sentiment)
            print()
            s.append(analysis.sentiment.polarity)
            twe.append(i.text)
            polarity = polarity + analysis.sentiment.polarity  # adding up polarities to find the average later

            if(analysis.sentiment.polarity>0):
                positive=positive+1
            elif(analysis.sentiment.polarity<0):
                negative=negative+1
            else:
                neutral=neutral+1
        
        

        #for w in self.tweetText:
         #   cw.writerow([w])
        #fo.close()

        dict={'senti':s,'tweets':twe}
        import pandas as pd
        caption=pd.DataFrame(dict)

        def clean(sentence):
            sentence=re.sub('[^a-zA-Z]',' ',sentence)
            sentence=sentence.lower()
            sentence=re.sub(r'#','',sentence)
            return(sentence)
        caption["pure_tweets"]=caption["tweets"].apply(clean)
        list1=caption['pure_tweets'].values
        from sklearn.feature_extraction.text import CountVectorizer
        cv=CountVectorizer(max_features=80)
        x=cv.fit_transform(list1)
        caption['senti'].values[caption['senti'].values > 0] = 1
        caption['senti'].values[caption['senti'].values < 0] = 2
        caption['senti'].values[caption['senti'].values == 0] = 3

        y=caption['senti']
        
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 50)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        result123 = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(result123)
        result1234 = classification_report(y_test, y_pred)
        print("Classification Report:",)
        print (result1234)
        result2345 = accuracy_score(y_test,y_pred)
        print("Accuracy:",result2345)

        positive = self.percentage(positive, number_of_terms)
        negative = self.percentage(negative, number_of_terms)
        neutral = self.percentage(neutral, number_of_terms)
        
        polarity=polarity/number_of_terms
        print("How people are reacting on " + word_search + " by analyzing " + str(number_of_terms) + " tweets.")
        print()
        print("Detailed Report: ")
        print(str(positive) + "% people thought it was positive")
        print(str(negative) + "% people thought it was negative")
        print(str(neutral) + "% people thought it was neutral")       


        self.bar_plot(positive,negative,neutral,word_search,number_of_terms)

    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    def percentage(self, x, y):
        temp = (float(x) / float(y))*100
        return temp

    def bar_plot(self,positive,negative,neutral,word_search,number_of_terms):
        objects=('positive','negative','neutral')
        performance=(positive,negative,neutral)
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Percentage')
        plt.xlabel('Sentiments')
        plt.show()

        names='POSITIVE','NEGATIVE','NEUTRAL'
        size=[positive,negative,neutral]
        
        # create a figure and set different background
        fig = plt.figure()
        fig.patch.set_facecolor('black')
        
        # Change color of text
        plt.rcParams['text.color'] = 'white'
        
        # Create a circle for the center of the plot
        my_circle=plt.Circle( (0,0), 0.7, color='black')
        
        # Pieplot + circle on it
        plt.pie(size, labels=names)
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.show()


        plt.plot([positive,negative,neutral],'ks-',mec='y',mew=2,ms=15)
        plt.ylabel('Percentage')
        plt.xlabel('Sentiments')
        mpld3.enable_notebook()
        mpld3.show()
    
    

        

    
    
    

# %%

if __name__== "__main__":
    sa = SA()
    sa.Data()

    


# %%
