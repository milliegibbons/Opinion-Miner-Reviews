import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()

import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS

from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

import re
import en_core_web_sm
nlp = en_core_web_sm.load()
from apyori import apriori
from mlxtend.frequent_patterns import apriori

import collections

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def opinion_mining(file):
    
    #PREPROCESSING
    name = file.split('.txt',1) #product name
    product_name = name[0] 
    
    f = open(file,encoding="utf8", errors='ignore')
    merged = ""
    
    manual_features = [] 

    for sent in f.readlines():
        if sent.startswith("*") or sent.startswith("[t]") or sent.startswith("  ") :
            pass
        elif sent.startswith("#"):
            new = re.sub(r'.*]##', "", sent) #remove everything before the ##
            new = new.replace('##',"") #remove everything before the ##
            merged += "\n" + new.strip() #put each sentence on a new line
        else:
            manual_feature = sent.split('##',1) #remove the manually inputted features
            manual_features.append(manual_feature[0]) #store as a list


    reviews = []
    for a_line in merged.split("\n"): #iterate through each line

        review = []
        doc = nlp(a_line) #turn the sentence into chunks 

        for chunk in doc.noun_chunks: #find the noun chunks in each sentence/ line
            feature = ""
            for token in chunk:
                if not token.is_stop: #take out the stop words
                    feature += str(token.lemma_) + " " #lemmatize the remaining words 
            if feature != "": #if the token was just a stop word , remove the blank space
                review.append(feature) #store as feature 
        reviews.append(review)


    df = pd.DataFrame(reviews[0:20]) #store features are dataframe 

    
    #FIND FEATURES using Apriori algorithm 
    te = TransactionEncoder()  #transform the reviews data frame into transaction encoder
    te_array = te.fit(reviews).transform(reviews)
    df = pd.DataFrame(te_array, columns=te.columns_)
    df

    apriori(df, min_support=0.01) #implement aprior, with minimum support as 1% as in the text
    freq_items = apriori(df, min_support=0.01, use_colnames=True)

    sorted_freq_items = freq_items.sort_values('support',ascending=False) #Sort items with highest support 

    features = [list(x) for x in sorted_freq_items['itemsets'].tolist()] #turn into a list 

    sorted_features = [] #create list of top 5 features 
    for feature in features[:5]:
        for f in feature:
            sorted_features.append(f.rstrip())

    all_features = [] #create list of all features 
    for feature in features:
        for f in feature:
            all_features.append(f.rstrip())
    
    #FIND OPINIONS 
    adjective_list = []
    for a_line in merged.split("\n"): #iterate through each line

        doc = nlp(a_line)
        doc = " ".join(token.lemma_ for token in doc if not token.is_stop) #preprocess 

        for word in sorted_features: #find reviews that contain feature 
            if word in doc:

                feature_opinion = []
                doc_nlp = (nlp(doc))
                 
                for token in doc_nlp:
                     if token.pos_=='ADJ': #find adjectives in sentences 
                        adjective_list.append(token) #list adjectives for each feature 

    #list of opinion words with their sentiment orientation 
    seed_list_sentiment = [['super','positive'],['love','positive'],['great','positive'],['perfect','positive'],['fantastic','positive'],['nice','positive'],['cool','positive'],['love','positive'],['positive','positive'],
                ['bad','negative'],['dull','negative'],['hate','negative'],['rubbish','negative'], ['dislike','negative'],['dreadful','negative'],['gross','negative'],['nasty','negative'],['poor','negative']
            
                    ]
    seed_list = []
    def orientation_prediction(adjective_list, seed_list):
        size_2 = []
        while len(seed_list) != size_2:
            orientation_search(adjective_list, seed_list)
            size_2 = len(seed_list)

    def orientation_search(adjective_list, seed_list):
        for i in range(len(seed_list_sentiment)):
            seed_list.append(seed_list_sentiment[i][0])

        for word in adjective_list: #iterate through adjective list 
            if str(word) in seed_list:
                pass
            else:
                synonyms = []
                antonyms = []

                for syn in wordnet.synsets(str(word)): #find synonyms and antonyms 
                    for l in syn.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                             antonyms.append(l.antonyms()[0].name())

                for syn in synonyms: #if synonoms are in seed list, add the orientation to that adjective and add too list
                    if syn in seed_list:
                        for i in range(len(seed_list)):
                            if syn == seed_list_sentiment[i][0]:
                                orientation = seed_list_sentiment[i][1]

                                if str(word) not in seed_list:
                                    seed_list_sentiment.append([str(word),orientation])
                                    seed_list.append(str(word))

                for syn in antonyms: #if antonyms are in seed list, add the opposite orientation to that adjective and add too list
                    if syn in seed_list:
                        for i in range(len(seed_list)):
                            if syn == seed_list_sentiment[i][0]:
                                orientation = seed_list_sentiment[i][1]
                                if seed_list_sentiment[i][1] == 'positive':
                                    orientation = 'negative'

                                if seed_list_sentiment[i][1] == 'negative':
                                    orientation = 'positive'

                                if str(word) not in seed_list:
                                    seed_list_sentiment.append([str(word),orientation])
                                    seed_list.append(str(word))


    orientation_prediction(adjective_list, seed_list)
 
    sorted_features = []
    for feature in features[:5]:
        for f in feature:
            sorted_features.append(f.rstrip())

    #SENTIMENT ANALYSIS 
    final = []
    for a_line in merged.split("\n"):  #for each feature find a review that has both a feature and an opinion 

        if any(feature in a_line for feature in sorted_features):
            if any(opinion in a_line for opinion in seed_list):

                for feature in sorted_features:
                    for opinion in seed_list:
                        if feature in a_line and opinion in a_line:

                            for i in range(len(seed_list_sentiment)):
                                if opinion == seed_list_sentiment[i][0]:
                                           sentiment = seed_list_sentiment[i][1]
                            final.append([feature,sentiment, a_line]) #store each feature, with its sentiment and its review 
                            
    output = [['feature','','','','','positive reviews','negative reviews']]


    for i in range(0,5):
        top_feature = sorted_features[i]

        positive = 0
        negative = 0

        pos_reviews = []
        neg_reviews = []
        for entry in final:

            feature,sentiment, a_line = entry

            if feature == top_feature:
                if sentiment == "positive":
                    positive += 1
                    pos_reviews.append(a_line)
                if sentiment == "negative":
                    negative += 1
                    neg_reviews.append(a_line)

        output.append([top_feature,'positive',positive,'negative',negative,pos_reviews, neg_reviews])
    
    print(product_name)
    print(pd.DataFrame(output))

    #Process manual features 
    manual_feature = []
    manual_sentiment = []
    for manual in manual_features:
        if '+' in manual:
            split_string = manual.split('[',1)
            feature = split_string[0]
            manual_sentiment.append([feature,'positive'])
        if '-' in manual:
            split_string = manual.split('[',1)
            feature = split_string[0]
            manual_sentiment.append([feature,'negative'])

    for manual in manual_features:
        split_string = manual.split('[',1)
        feature = split_string[0]
        manual_feature.append(feature)

    word_counts = collections.Counter(manual_feature)

    word_counts_top = word_counts.most_common((len(all_features)))

    sorted_features_all = sorted(all_features)

    sorted_manual_features= (sorted(list(set(manual_feature))))

    #Find intersection between found features and manual features 
    intersection = set(sorted_manual_features)
    intersection = [x for x in sorted_features_all if x in intersection]

    mylist = list(dict.fromkeys(intersection)) #remove duplicates
    intersection = mylist

    manual_output = []
    for i in range(0,5):
        manualfeature = word_counts_top[i][0]

        manual_positive = 0
        manual_negative = 0

        for entry in manual_sentiment:

            feature,sentiment = entry

            if feature == manualfeature:
                if sentiment == "positive":
                    manual_positive += 1
                if sentiment == "negative":
                    manual_negative += 1

        manual_output.append([manualfeature,manual_positive,manual_negative])

    print(" ")
    print(" ")
    
    sorted_features = []
    for feature in features[:5]:
        for f in feature:
            sorted_features.append(f.rstrip())

    print("Top 5 mined features:","        ", sorted_features)
    
    top_manual_feature = []
    for i in range(0,5):
        top_manual_feature.append(word_counts_top[i][0])

    print("Top 5 manually found features:", top_manual_feature)
     
    #FEATURE EXTRACTION PRECISION AND RECALL
    TP = len(intersection) #number in both mine and manual
    TN = 0
    FP = len(sorted_features_all)- len(intersection)#values i found that arent in manual
    FN = len(sorted_manual_features)-len(intersection) #manual features that i diddnt find 

    #FEATURE EXTRACTION PRECISION AND RECALL
    recall = TP / (TP + FN)
    print('recall:',"  ",recall)
    
    #FEATURE EXTRACTION PRECISION AND RECALL
    precision = TP / (TP + FP)
    print('precision:',precision)
    
    print(" ")
    print(" ")
    
    manual_output = []
    output = [['feature','','mined','manual','','mined','manual']]
    
    #Sentiment analysis of features in intersection 
    if len(intersection) <5:
        intersect_range = len(intersection)
    else: 
        intersect_range = 5
        
    for i in range(0,intersect_range):
            intersection_feature = intersection[i]
            intersection_positive = 0
            intersection_negative = 0
            intersection_positive1 = 0
            intersection_negative1 = 0

            for entry in manual_sentiment:

                feature,sentiment = entry

                if feature == intersection_feature:
                    if sentiment == "positive":
                        intersection_positive += 1
                    if sentiment == "negative":
                        intersection_negative += 1

            manual_output.append([intersection_feature,'positive',intersection_positive,'negative',intersection_negative])
            pos_reviews = []
            neg_reviews = []
            for entry in final:

                feature,sentiment, a_line = entry

                if feature == intersection_feature:
                    if sentiment == "positive":
                        intersection_positive1 += 1
                        pos_reviews.append(a_line)
                    if sentiment == "negative":
                        intersection_negative1 += 1
                        neg_reviews.append(a_line)

            output.append([intersection_feature,'positive',intersection_positive1, intersection_positive,'negative', intersection_negative1,intersection_negative])

    print("comparison of intersection features")
    print(pd.DataFrame(output))
    
    #FEATURE EXTRACTION PRECISION AND RECALL
    TP =  #number in both mine and manual
    TN = 0
    FP = len(sorted_features_all)- len(intersection)#values i found that arent in manual
    FN = len(sorted_manual_features)-len(intersection) #manual features that i diddnt find 

    #FEATURE EXTRACTION PRECISION AND RECALL
    recall = TP / (TP + FN)
    print('recall:',"  ",recall)
    
    #FEATURE EXTRACTION PRECISION AND RECALL
    precision = TP / (TP + FP)
    print('precision:',precision)
