import TRECCASTeval as trec
import numpy as np
import pprint

import numpy as np

import OpenSearchSimpleAPI as osearch
import pprint as pp

import bm25s

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

pp = pprint.PrettyPrinter(indent=4)

test_bed = trec.ConvSearchEvaluation()

#print()
#print("========================================== Training conversations =====")
topics = {}
for topic in test_bed.train_topics:
    conv_id = topic['number']

    if conv_id not in (1, 2, 4, 7, 15, 17,18,22,23,24,25,27,30):
        continue

    #print()
    #print(conv_id, "  ", topic['title'])

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        #print(topic_turn_id, utterance)
        topics[topic_turn_id] = utterance

#print()
#print("========================================== Test conversations =====")
for topic in test_bed.test_topics:
    conv_id = topic['number']

    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue


    #print(conv_id, "  ", topic['title'])

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        #print(topic_turn_id, utterance)
        topics[topic_turn_id] = utterance

test_bed.test_relevance_judgments


opensearch = osearch.OSsimpleAPI()


numdocs = 100
test_query = topics['40_1']

opensearch_results = opensearch.search_body(test_query, numDocs = numdocs)


# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def expand_query(query):
    # list of synonms
    synonyms = {
        "music": ["song","melody", "opera", "tune"],
        "origin": ["ancestor", "ancestry", "connection","root", "source"],
        "popular": ["attractive","beloved","famous"]
    }
    
    expanded_query = query
    for word in query.split():
        if word in synonyms:
            expanded_query += ' ' + ' '.join(synonyms[word]) + 'music music music music'
    
    return expanded_query

# Build the corpus
corpus = []
for index, row in opensearch_results.iterrows():
    doc_id = row['_id']
    doc_body = opensearch.get_doc_body(doc_id)
 #   processed_doc = preprocess_text(doc_body) 
    corpus.append(doc_body)


query = test_query
processed_query = preprocess_text(query)
expanded_query = expand_query(processed_query)
print('Unprocessed query: ' + query)
print('Processed: ' + processed_query)
print('Expanded Query:', expanded_query)

# tokenize and index
tokenized_query = bm25s.tokenize(expanded_query)
print("Tokenized Query:", tokenized_query)

retriever = bm25s.BM25(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))

k = 100
results, scores = retriever.retrieve(tokenized_query, k=k)

for i in range(len(results[0])):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

