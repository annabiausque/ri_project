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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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
test_query = topics['34_1']
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


opensearch_results = opensearch.search_body(test_query, numDocs = numdocs)
print(opensearch_results)

opensearch.doc_term_vectors('CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

opensearch.termvectors_JSON(doc_id='CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

opensearch.get_doc_body('CAR_c370ef5df77de117ff7d02c4b64b52f5bae9abc9')

example_doc = 'The NeverEnding Story III: Escape from Fantasia (also known as: The NeverEnding Story III: Return to Fantasia) is a 1994 film and the second sequel to the fantasy film The NeverEnding Story (following the first sequel The NeverEnding Story II: The Next Chapter). It starred Jason James Richter as the principal character Bastian Bux'
opensearch.query_terms(example_doc,'standard')

opensearch.analyzer(analyzer="standard", query=example_doc)

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # stemming
    text = ' '.join([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in text.split()])
    return text

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if (lemma.name().lower() != word):
                synonyms.append(lemma.name())
        if len(synonyms) >= 3:
            break
    return synonyms

#def expand_query(query):
    #list of synonms    
    #expanded_query = query
    #for word in query.split():
    #    synonyms = get_synonyms(word)
    #    expanded_query += ' ' + ' '.join(synonyms)
    #return expanded_query

# Build the corpus
corpus = []
for index, row in opensearch_results.iterrows():
    doc_id = row['_id']
    doc_body = opensearch.get_doc_body(doc_id)
    processed_doc = preprocess_text(doc_body) 
    corpus.append(doc_body)


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

