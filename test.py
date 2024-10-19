import TRECCASTeval as trec
import numpy as np
import pprint
import pandas as pd
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

# Initialize stop words and stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

print()
print("========================================== Training conversations =====")
topics = {}
for topic in test_bed.train_topics:
    conv_id = topic['number']

    if conv_id not in (1, 2, 4, 7, 15, 17,18,22,23,24,25,27,30):
        continue

    print()
    print(conv_id, "  ", topic['title'])

    previous_query_tokenized = ''
    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        updated_utterance = previous_query_tokenized + utterance
        previous_query_tokenized += preprocess_text(utterance) + ' '
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        print(topic_turn_id, updated_utterance)
        topics[topic_turn_id] = updated_utterance

print()
print("========================================== Test conversations =====")
for topic in test_bed.test_topics:
    conv_id = topic['number']

    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue


    #print(conv_id, "  ", topic['title'])

    previous_query_tokenized = ''
    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        updated_utterance = previous_query_tokenized + utterance
        previous_query_tokenized += preprocess_text(utterance) + ' '
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        print(topic_turn_id, updated_utterance)
        topics[topic_turn_id] = updated_utterance

test_bed.test_relevance_judgments


opensearch = osearch.OSsimpleAPI()


numdocs = 100
test_query = topics['34_3']
'''Testing to use opensearch similarity search options
opensearch.client.indices.close(index=opensearch.index_name)
opensearch.client.indices.put_settings(index=opensearch.index_name, body={
    "settings": {
        "index": {
            "similarity": {
                "default": {
                    "type": "LMJelinekMercer",
                    "lambda": 0.7  # Smoothing parameter for Jelinek-Mercer
                }
            }
        }
    }
})
opensearch.client.indices.open(index=opensearch.index_name)

#Check if settings are updating correctly
index_settings = opensearch.client.indices.get_settings(index=opensearch.index_name)
print("Updated Index Settings:")
pp.pprint(index_settings)
'''

opensearch_results = opensearch.search_body(test_query, numDocs = numdocs)
print(opensearch_results)

opensearch.doc_term_vectors('CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

opensearch.termvectors_JSON(doc_id='CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

opensearch.get_doc_body('CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

example_doc = 'The NeverEnding Story III: Escape from Fantasia (also known as: The NeverEnding Story III: Return to Fantasia) is a 1994 film and the second sequel to the fantasy film The NeverEnding Story (following the first sequel The NeverEnding Story II: The Next Chapter). It starred Jason James Richter as the principal character Bastian Bux'
opensearch.query_terms(example_doc,'standard')

opensearch.analyzer(analyzer="standard", query=example_doc)

#def expand_query(query):
    #list of synonms    
    #expanded_query = query
    #for word in query.split():
    #    synonyms = get_synonyms(word)
    #    expanded_query += ' ' + ' '.join(synonyms)
    #return expanded_query

# Build the corpus
corpus = []
doc_ids = []
content_to_id = {}  
for index, row in opensearch_results.iterrows():
    doc_id = row['_id']
    doc_body = opensearch.get_doc_body(doc_id)
    corpus.append(doc_body)
    doc_ids.append(doc_id)
    content_to_id[doc_body] = doc_id  

query = test_query
processed_query = preprocess_text(query)
#expanded_query = expand_query(processed_query)
print('Unprocessed query: ' + query)
print('Processed: ' + processed_query)

# tokenize and index
tokenized_query = bm25s.tokenize(processed_query)
print("Tokenized Query:", tokenized_query)

retriever = bm25s.BM25(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))

k = 100
results, scores = retriever.retrieve(tokenized_query, k=k)

for i in range(len(results[0])):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# Making the tuple (#turn; query; top N passages), with N = 3
data = []

for turn in topics:
    query = topics[turn]
    processed_query = preprocess_text(query)
    tokenized_query = bm25s.tokenize(processed_query)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))
    k = 3
    results, scores = retriever.retrieve(tokenized_query, k=k)
    best_docs = []
    for i in range(len(results[0])):
        doc, score = results [0, i], scores[0, i]
        best_docs.append(content_to_id[doc])
    data.append({'turn': turn, 'query': query, 'top passages': best_docs})
df = pd.DataFrame(data)
print(df)
    