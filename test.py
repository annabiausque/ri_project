import TRECCASTeval as trec
import numpy as np
import pprint

import numpy as np

import OpenSearchSimpleAPI as osearch
import pprint as pp

import bm25s


pp = pprint.PrettyPrinter(indent=4)

test_bed = trec.ConvSearchEvaluation()

print()
print("========================================== Training conversations =====")
topics = {}
for topic in test_bed.train_topics:
    conv_id = topic['number']

    if conv_id not in (1, 2, 4, 7, 15, 17,18,22,23,24,25,27,30):
        continue

    print()
    print(conv_id, "  ", topic['title'])

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        print(topic_turn_id, utterance)
        topics[topic_turn_id] = utterance

print()
print("========================================== Test conversations =====")
for topic in test_bed.test_topics:
    conv_id = topic['number']

    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue

    print()
    print(conv_id, "  ", topic['title'])

    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d'% (conv_id, turn_id)
        
        print(topic_turn_id, utterance)
        topics[topic_turn_id] = utterance

test_bed.test_relevance_judgments


opensearch = osearch.OSsimpleAPI()


numdocs = 10
test_query = topics['40_1']
opensearch_results = opensearch.search_body(test_query, numDocs = numdocs)
print(opensearch_results)

opensearch.doc_term_vectors('CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

opensearch.termvectors_JSON(doc_id='CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

opensearch.get_doc_body('CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

example_doc = 'The NeverEnding Story III: Escape from Fantasia (also known as: The NeverEnding Story III: Return to Fantasia) is a 1994 film and the second sequel to the fantasy film The NeverEnding Story (following the first sequel The NeverEnding Story II: The Next Chapter). It starred Jason James Richter as the principal character Bastian Bux'
opensearch.query_terms(example_doc,'standard')

opensearch.analyzer(analyzer="standard", query=example_doc)

# pip install bm25s

# Create your corpus here
test_corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

# Create the BM25 model and index the corpus
retriever = bm25s.BM25(corpus=test_corpus)
retriever.index(bm25s.tokenize(test_corpus))

# Query the corpus and get top-k results
query = "does the fish purr like a cat?"
k = 4
test_results, scores = retriever.retrieve(bm25s.tokenize(query), k=k)

# Let's see what we got!
for i in range(len(test_results[0])):
    doc, score = test_results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

import re

def preprocess_text(text):
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Build the corpus
corpus = []
for index, row in opensearch_results.iterrows():
    doc_id = row['_id']
    doc_body = opensearch.get_doc_body(doc_id)
    processed_doc = preprocess_text(doc_body)  # Preprocess the document
    corpus.append(processed_doc)

tokenized_query = bm25s.tokenize(query)
print("Tokenized Query:", tokenized_query)

retriever = bm25s.BM25(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))
query = test_query
processed_query = preprocess_text(query)
print('Unprocessed query: ' + query)
print('processed: ' + processed_query)

k = 4

results, scores = retriever.retrieve(bm25s.tokenize(processed_query), k=k)
for i in range(len(results[0])):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")