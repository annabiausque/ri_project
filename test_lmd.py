import TRECCASTeval as trec
import numpy as np
import pprint
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import OpenSearchSimpleAPI as osearch
import bm25s

nltk.download('stopwords')

pp = pprint.PrettyPrinter(indent=4)


test_bed = trec.ConvSearchEvaluation()


topics = {}
for topic in test_bed.train_topics:
    conv_id = topic['number']
    if conv_id not in (1, 2, 4, 7, 15, 17, 18, 22, 23, 24, 25, 27, 30):
        continue
    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d' % (conv_id, turn_id)
        topics[topic_turn_id] = utterance

for topic in test_bed.test_topics:
    conv_id = topic['number']
    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue
    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d' % (conv_id, turn_id)
        topics[topic_turn_id] = utterance


opensearch = osearch.OSsimpleAPI()

numdocs = 100
test_query = topics['40_1']


opensearch_results = opensearch.search_body(test_query, numDocs=numdocs)


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def expand_query(query):
    expanded_query = query + ' music music music music'
    return expanded_query


query = test_query
processed_query = preprocess_text(query)
expanded_query = expand_query(processed_query)
print('Unprocessed query: ' + query)
print('Processed: ' + processed_query)
print('Expanded Query:', expanded_query)

corpus = []
doc_ids = []
content_to_id = {}  
for index, row in opensearch_results.iterrows():
    doc_id = row['_id']
    doc_body = opensearch.get_doc_body(doc_id)
    corpus.append(doc_body)
    doc_ids.append(doc_id)
    content_to_id[doc_body] = doc_id  

# BM25 retrieval
tokenized_query = bm25s.tokenize(expanded_query)
print("Tokenized Query:", tokenized_query)

retriever = bm25s.BM25(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))

k = 100
bm25_results, bm25_scores = retriever.retrieve(tokenized_query, k=k)


bm25_doc_ids = [content_to_id[doc] for doc in bm25_results[0] if doc in content_to_id]

# LMDRetriever
class LMDRetriever:
    def __init__(self, opensearch, corpus_ids):
        self.opensearch = opensearch
        self.corpus_ids = corpus_ids
        self.corpus_length = 0
        self.doc_count = len(corpus_ids)
        self.index = self.build_index()
        self.collection_frequency = self.build_collection_frequency()
        self.mu = self.calculate_mu()
    
    def calculate_mu(self):
        avg_doc_length = self.corpus_length / self.doc_count
        return 0.1 * avg_doc_length

    def build_index(self):
        index = {}
        for doc_id in self.corpus_ids:
            term_vectors = self.opensearch.doc_term_vectors(doc_id)
            if term_vectors:
                terms = term_vectors[3]
                for term, stats in terms.items():
                    if term not in index:
                        index[term] = {}
                    index[term][doc_id] = stats[0]
                    self.corpus_length += stats[0]
        return index

    def build_collection_frequency(self):
        collection_frequency = {}
        for doc_id in self.corpus_ids:
            term_vectors = self.opensearch.doc_term_vectors(doc_id)
            if term_vectors:
                terms = term_vectors[3]
                for term, stats in terms.items():
                    if term not in collection_frequency:
                        collection_frequency[term] = 0
                    collection_frequency[term] += stats[2]
        return collection_frequency

    def score(self, query, doc_id):
        score = 1.0
        term_vectors = self.opensearch.doc_term_vectors(doc_id)
        if term_vectors:
            terms = term_vectors[3]
            doc_length = sum([stats[0] for stats in terms.values()])
            for term in query.split():
                tf = terms.get(term, [0])[0]
                cf = self.collection_frequency.get(term, 0)
                p_ml = cf / self.corpus_length
                p_lmd = (tf + self.mu * p_ml) / (doc_length + self.mu)
                if p_lmd > 0:
                    score *= p_lmd
        return score

    def retrieve(self, query, k):
        scores = []
        for doc_id in self.corpus_ids:
            score = self.score(query, doc_id)
            scores.append((doc_id, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:k]

# LMDRetriever to rerank the BM25 results
lmd_retriever = LMDRetriever(opensearch=opensearch, corpus_ids=bm25_doc_ids)
reranked_results = lmd_retriever.retrieve(expanded_query, k=10)


for rank, (doc_id, score) in enumerate(reranked_results, start=1):
    doc_content = opensearch.get_doc_body(doc_id)
    print(f"Rank {rank} (score: {score:.6f}):\n{doc_content}\n")
