import TRECCASTeval as trec
import numpy as np
import pprint

import numpy as np

import OpenSearchSimpleAPI as osearch
import pprint as pp

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

pp = pprint.PrettyPrinter(indent=4)

test_bed = trec.ConvSearchEvaluation()

# Préparer les conversations d'entraînement
topics = {}
for topic in test_bed.train_topics:
    conv_id = topic['number']

    if conv_id not in (1, 2, 4, 7, 15, 17,18,22,23,24,25,27,30):
        continue

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d' % (conv_id, turn_id)
        topics[topic_turn_id] = utterance

# Préparer les conversations de test
for topic in test_bed.test_topics:
    conv_id = topic['number']

    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d' % (conv_id, turn_id)
        topics[topic_turn_id] = utterance

test_bed.test_relevance_judgments

opensearch = osearch.OSsimpleAPI()

numdocs = 100
test_query = topics['40_1']

opensearch_results = opensearch.search_body(test_query, numDocs=numdocs)

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

    expanded_query = query + ' music music music music'
    
    return expanded_query

# Build the corpus
corpus = []
for index, row in opensearch_results.iterrows():
    doc_id = row['_id']
    doc_body = opensearch.get_doc_body(doc_id)
    corpus.append(doc_body)

query = test_query
processed_query = preprocess_text(query)
expanded_query = expand_query(processed_query)
print('Unprocessed query: ' + query)
print('Processed: ' + processed_query)
print('Expanded Query:', expanded_query)


class LMDRetriever:
    def __init__(self, corpus, mu=2000):
        self.corpus = corpus
        self.mu = mu
        self.index = self.build_index(corpus)
        self.corpus_length = sum(len(doc.split()) for doc in corpus)
        self.collection_frequency = self.build_collection_frequency(corpus)
    
    def build_index(self, corpus):
        index = {}
        for doc_id, doc in enumerate(corpus):
            for term in doc.split():
                if term not in index:
                    index[term] = {}
                if doc_id not in index[term]:
                    index[term][doc_id] = 0
                index[term][doc_id] += 1
        print(index)
        return index

    

    def build_collection_frequency(self, corpus):
        collection_frequency = {}
        for doc in corpus:
            for term in doc.split():
                if term not in collection_frequency:
                    collection_frequency[term] = 0
                collection_frequency[term] += 1
        return collection_frequency

    def score(self, query, doc_id):
        score = 0.0
        doc_length = len(self.corpus[doc_id].split())
        for term in query.split():
            tf = self.index.get(term, {}).get(doc_id, 0)
            cf = self.collection_frequency.get(term, 0)
            p_ml = cf / self.corpus_length
            p_lmd = (tf + self.mu * p_ml) / (doc_length + self.mu)
            score += np.log(p_lmd)
        return score

    def retrieve(self, query, k=5):
        scores = []
        for doc_id in range(len(self.corpus)):
            score = self.score(query, doc_id)
            scores.append((doc_id, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:k]

# Create an instance of LMDRetriever and perform retrieval
retriever = LMDRetriever(corpus=corpus)
results = retriever.retrieve(expanded_query, k=4)

# Display the results
for rank, (doc_id, score) in enumerate(results, start=1):
    doc = corpus[doc_id]
    print(f"Rank {rank} (score: {score:.2f}): {doc}")
