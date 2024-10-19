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
test_query = topics['77_1']


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
    expanded_query = query 
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
    def __init__(test_bed, opensearch, corpus_ids):
        test_bed.opensearch = opensearch
        test_bed.corpus_ids = corpus_ids
        test_bed.corpus_length = 0
        test_bed.doc_count = len(corpus_ids)
        test_bed.index = test_bed.build_index()
        test_bed.collection_frequency = test_bed.build_collection_frequency()
        test_bed.mu = test_bed.calculate_mu()
    
    def calculate_mu(test_bed):
        avg_doc_length = test_bed.corpus_length / test_bed.doc_count
        return 0.1 * avg_doc_length

    def build_index(test_bed):
        index = {}
        for doc_id in test_bed.corpus_ids:
            term_vectors = test_bed.opensearch.doc_term_vectors(doc_id)
            if term_vectors:
                terms = term_vectors[3]
                for term, stats in terms.items():
                    if term not in index:
                        index[term] = {}
                    index[term][doc_id] = stats[0]
                    test_bed.corpus_length += stats[0]
        return index

    def build_collection_frequency(test_bed):
        collection_frequency = {}
        for doc_id in test_bed.corpus_ids:
            term_vectors = test_bed.opensearch.doc_term_vectors(doc_id)
            if term_vectors:
                terms = term_vectors[3]
                for term, stats in terms.items():
                    if term not in collection_frequency:
                        collection_frequency[term] = 0
                    collection_frequency[term] += stats[2]
        return collection_frequency

    def score(test_bed, query, doc_id):
        score = 1.0
        term_vectors = test_bed.opensearch.doc_term_vectors(doc_id)
        if term_vectors:
            terms = term_vectors[3]
            doc_length = sum([stats[0] for stats in terms.values()])
            for term in query.split():
                tf = terms.get(term, [0])[0]
                cf = test_bed.collection_frequency.get(term, 0)
                p_ml = cf / test_bed.corpus_length
                p_lmd = (tf + test_bed.mu * p_ml) / (doc_length + test_bed.mu)
                if p_lmd > 0:
                    score *= p_lmd
        return score

    def retrieve(test_bed, query, k):
        scores = []
        for doc_id in test_bed.corpus_ids:
            score = test_bed.score(query, doc_id)
            scores.append((doc_id, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:k]

# LMDRetriever to rerank the BM25 results
lmd_retriever = LMDRetriever(opensearch=opensearch, corpus_ids=bm25_doc_ids)
reranked_results = lmd_retriever.retrieve(expanded_query, k=10)

data = []
best_docs = []
for rank, (doc_id, score) in enumerate(reranked_results, start=1):
    doc_content = opensearch.get_doc_body(doc_id)
    best_docs.append(doc_id)
    print(f"Rank {rank} (score: {score:.6f}):\n{doc_content}\n")

data.append({"turn" : "77_1", "query" : query, '_id' : best_docs })
result_df = pd.DataFrame(data)


total_retrieved_docs = len(reranked_results)

aux = test_bed.train_relevance_judgments.loc[test_bed.train_relevance_judgments['topic_turn_id'] == "77_1"]
rel_docs = aux.loc[aux['rel'] != 0]

if np.size(aux) == 0 :
        aux = test_bed.test_relevance_judgments.loc[test_bed.test_relevance_judgments['topic_turn_id'] == "77_1"]
    
ground_truth = (aux.loc[aux['rel'] != 0]).sort_values(by='rel', ascending=False)

print(ground_truth)

metrics_LMD = []
true = 0


for rank, (doc_id, score) in enumerate(reranked_results[:10], start=1):
    relevance_level = 0  
    for index, row in ground_truth.iterrows():
        if doc_id == row["docid"]:
            relevance_level = row["rel"]  
            break 
    
    metrics_LMD.append({"rank": rank, "id": doc_id, "relevant": relevance_level})

