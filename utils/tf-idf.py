
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

# just the 1st time
# nltk.download('punkt')

class Collection:
    """
        Class representing a document collection.
        Methods exposed for adding, removing, calculate term frequency and get rank
    """

    def __init__(self):
        self.docs = [] # Original documents
        self.preprocessed_docs = [] # Documents after preprocess()
        self.processed_docs = [] # Documents after preprocess() and tokenization
        self.updated = True # True iff td-idf has been calculated on all the collection
        self.ndocs = 0 # Counter of documents number

        self.tf_idf = {} # tf-idf dictionary

    def preprocess(self, doc):
        def convert_lower_case(data):
            return np.char.lower(data)
        
        def remove_stop_words(data):
            stop_words = stopwords.words('english')
            words = word_tokenize(str(data))
            new_text = ""
            for w in words:
                if w not in stop_words and len(w) > 1:
                    new_text = new_text + " " + w
            return new_text

        def remove_punctuation(data):
            symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
            for i in range(len(symbols)):
                data = np.char.replace(data, symbols[i], ' ')
                data = np.char.replace(data, "  ", " ")
            data = np.char.replace(data, ',', '')
            return data

        def remove_apostrophe(data):
            return np.char.replace(data, "'", "")

        def stemming(data):
            stemmer= PorterStemmer()
            tokens = word_tokenize(str(data))
            new_text = ""
            for w in tokens:
                new_text = new_text + " " + stemmer.stem(w)
            return new_text
        
        def convert_numbers(data):
            tokens = word_tokenize(str(data))
            new_text = ""
            for w in tokens:
                try:
                    w = num2words(int(w))
                except:
                    a = 0
                new_text = new_text + " " + w
            new_text = np.char.replace(new_text, "-", " ")
            return new_text

        doc = convert_lower_case(doc)
        doc = remove_punctuation(doc) #remove comma seperately
        doc = remove_apostrophe(doc)
        doc = remove_stop_words(doc)
        doc = convert_numbers(doc)
        doc = stemming(doc)
        doc = remove_punctuation(doc)
        doc = convert_numbers(doc)
        doc = stemming(doc) #needed again as we need to stem the words
        doc = remove_punctuation(doc) #needed again as num2word is giving few hypens and commas fourty-one
        doc = remove_stop_words(doc) #needed again as num2word is giving stop words 101 - one hundred and one

        return doc


    def addDocument(self, doc):
        """
        Adds doc to the document list (docs)
        Processes doc and adds:
        - the preprocessed document to preprocessed_docs
        - the processed document to processed_docs
        """
        # Storing original document
        self.docs.append(doc)

        # Preprocessing and storing
        preprocessed_doc = self.preprocess(doc)
        self.preprocessed_docs.append(preprocessed_doc)

        # Tokenizing and storing
        self.processed_docs.append(word_tokenize(str(preprocessed_doc)))

        self.ndocs += 1 # Updating docs counter
        self.updated = False # tf-idf on the collection is not updated since a document has just been added


    def addDocuments(self, docs):
        """
        For each doc in docs, it calls addDocument
        """
        for doc in docs:
            self.addDocument(doc)

    def getCollection(self):
        """
        Returns:
        - List of original documents
        - List of preprocessed documents
        - List of processed documens
        """
        return (self.docs, self.preprocessed_docs, self.processed_docs)

    def df(self):
        """
        Computes the Document Frequency for each term and stores it in self.DF
        """
        self.DF = {}
        for i in range(self.ndocs):
            tokens = self.processed_docs[i]
            for w in tokens:
                try:
                    self.DF[w].add(i)
                except:
                    self.DF[w] = {i}

        for i in self.DF:
            self.DF[i] = len(self.DF[i])

    def doc_freq(self, word):
        """
        Returns the Document Frequency of a given Term
        """
        c = 0
        try:
            c = self.DF[word]
        except:
            pass
        return c

    def tfidf(self, pair = None):
        """
        Computes the tf idf dictionary and stores it in self.tf_idf
        If parameter pair != None, returns the self.tf_idf entry for such pair, if any 
        """
        # If no documents have been added to the collection from the last tfidf computation
        if(self.updated): 
            return

        # Else
        self.df() # Compute df
        doc = 0
        self.tf_idf = {}

        for tokens in self.processed_docs:

            counter = Counter(tokens)
            words_count = len(tokens)
            
            for token in np.unique(tokens):
                # Term frequency
                tf = counter[token]/words_count
                # Doc frequency
                df = self.doc_freq(token)
                idf = np.log((self.ndocs+1)/(df+1))
                
                self.tf_idf[doc, token] = tf*idf

            doc += 1
        
        if(pair != None):
            return self.tf_idf[pair]
        
    
    def getTopDocs(self, k, query):
        """
        Computes the ranks of the documents according to the query
        Returns the top k matching documents
        """
        # Preprocessing
        preprocessed_query = self.preprocess(query)

        # Tokeinzing
        tokens = word_tokenize(str(preprocessed_query))

        # Printing
        print("\nQuery:", query)
        print("")
        print(tokens)

        # Obtaining score
        query_weights = {}

        for key in self.tf_idf:
            
            if key[1] in tokens:
                try:
                    query_weights[key[0]] += self.tf_idf[key]
                except:
                    query_weights[key[0]] = self.tf_idf[key]
        
        query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

        print("")
        
        l = []
        
        for i in query_weights[:k]:
            l.append(i[0])
        
        return l

    def getOriginalDocs(self, indices):
        docs = []
        append = docs.append
        for i in indices:
            append(self.docs[i])
        
        return docs



# Creating collection
c = Collection()

#Adding some documents (individually)
c.addDocument("The house is fantastic, we love it!")
c.addDocument("The house is enormous, but we dislike it!")
c.addDocument("Our house is  like yours but our is better, we hate yours!")
c.addDocument("The house is fantastic, we love it!")
c.addDocument("The house is very very gooood!")

# Adding some documents (testing function with list as parameter)
c.addDocuments(["This car is more beautiful than the previous house...", "in any case this is just an example message"])

# Getting docs, preprocessed docs and processed docs from collection
d, pre, p = c.getCollection()

print(d) # Ooriginal documents 
print(pre) # Preprocessed documents
print(p) # Tokenized documents

# Getting tfidf for (0, "hou") pair
print(c.tfidf((0, "hou"))) 

# Getting indices top k matching documents fot the given string
topK = c.getTopDocs(2, "The house is very good ") 

# Printing documents corresponding to such indices
print(c.getOriginalDocs(topK)) 