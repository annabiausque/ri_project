import TRECCASTeval as trec
import numpy as np
import pprint
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import OpenSearchSimpleAPI as osearch
import bm25s
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

nltk.download('stopwords')

pp = pprint.PrettyPrinter(indent=4)


test_bed = trec.ConvSearchEvaluation()

chosen_topic= 77
conversation = []
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
        if conv_id == chosen_topic :
            conversation.append({"conv_id" : conv_id, "turn_id" : turn_id, "utterance" : utterance})


for topic in test_bed.test_topics:
    conv_id = topic['number']

    if conv_id not in (31, 32, 33, 34, 37, 40, 49, 50, 54, 56, 58, 59, 61, 67, 68, 69, 75, 77, 78, 79):
        continue
    for turn in topic['turn']:
        turn_id = turn['number']
        utterance = turn['raw_utterance']
        topic_turn_id = '%d_%d' % (conv_id, turn_id)
        topics[topic_turn_id] = utterance
        if conv_id == chosen_topic :
            conversation.append({"conv_id" : conv_id, "turn_id" : turn_id, "utterance" : utterance})


opensearch = osearch.OSsimpleAPI()

numdocs = 100



def BM_retrieval(k):
    #metrics_df = pd.DataFrame(columns=['turn', 'query', '_id'])
    BM25data = []
    bm25_doc_ids = []
    for element in conversation :
        topic = str(chosen_topic)
        turn = str(element['turn_id'])
        utterance = topic + '_' + turn
   
        query = topics[utterance]
        opensearch_results = opensearch.search_body(query, numDocs = k)
        best_docs = []
        best_passages = []
        content_to_id = {}  
        for index, row in opensearch_results.iterrows():
            doc_id = row['_id']
            doc_body = opensearch.get_doc_body(doc_id)
            #new_row = {'turn': utterance, 'query': element['utterance'], '_id': doc_id}
            #metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
            best_passages.append(doc_body)
            best_docs.append(doc_id)
            content_to_id[doc_body] = doc_id  
        bm25_doc_ids.append([content_to_id[doc] for doc in best_passages if doc in content_to_id])
   
            
        BM25data.append({'turn': turn, 'query': element["utterance"], "expanded_query" : query,  'top passages': best_passages, '_id': best_docs})

    df = pd.DataFrame(BM25data)
   
    return df

print(BM_retrieval(3))

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

bm25_results = BM_retrieval(100)

#print(bm25_results)

#print('-------------------------------------------------------------------------------------')


reranked_df = pd.DataFrame(columns=["turn", "query", "_id"])


for index, row in bm25_results.iterrows():
   
    bm25_doc_ids = row["_id"] 
    turn = row["turn"]
    query = row["expanded_query"]

    # Créer une instance de LMDRetriever avec les doc_ids récupérés
    lmd_retriever = LMDRetriever(opensearch=opensearch, corpus_ids=bm25_doc_ids)

    # Reranking avec la méthode retrieve
    reranked_results = lmd_retriever.retrieve(query, k=100)

    # Extraire les passages du reranking
    top_N_passages = [doc_id for doc_id, score in reranked_results]

    # Créer un DataFrame temporaire pour la nouvelle ligne
    new_row = pd.DataFrame({
        "turn": [turn],
        "query": [query],
        "_id": [top_N_passages]
    })


    # Utiliser pd.concat pour ajouter la nouvelle ligne au DataFrame final
    reranked_df = pd.concat([reranked_df, new_row], ignore_index=True)

# Afficher le DataFrame final
print(reranked_df)


turns = []
LMD_ap_values = []
LMD_ndcg_values = []
LMD_precision_values = []
LMD_recall_values = []

print("LMD")

for index, row in reranked_df.iterrows():
    try:
    
        turn = f"77_{row['turn']}"  
        query = row['query']
        docs = row['_id']  

        result_df = pd.DataFrame({"_id": docs})

        p10, recall, ap, ndcg5 = test_bed.eval(result_df, turn)
        turns.append(turn)
        LMD_ap_values.append(ap)
        LMD_ndcg_values.append(ndcg5)
        LMD_precision_values.append(p10)
        LMD_recall_values.append(recall)

        print(f"Turn: {turn}")
        
    
        print(f"P@10: {p10}, Recall: {recall}, AP: {ap}, NDCG@5: {ndcg5}\n")

    except Exception as e:
  
        print(f"Erreur sur le tour {turn}: {e}")
        break  

print(turns,LMD_ap_values,LMD_ndcg_values,LMD_precision_values, LMD_recall_values)




filtered_turns = []
filtered_ap_values = []
filtered_ndcg_values = []
filtered_precision_values = []
filtered_recall_values = []

for i in range(len(turns)):

    if LMD_ap_values[i] != 0 or LMD_ndcg_values[i] != 0:
        filtered_turns.append(turns[i])
        filtered_ap_values.append(LMD_ap_values[i])
        filtered_ndcg_values.append(LMD_ndcg_values[i])

 
    if LMD_precision_values[i] != 0 or LMD_recall_values[i] != 0:
        filtered_precision_values.append(LMD_precision_values[i])
        filtered_recall_values.append(LMD_recall_values[i])



plt.figure(figsize=(10, 6))
plt.plot(filtered_turns, filtered_ap_values, marker='o', label='AP', linestyle='-')
plt.plot(filtered_turns, filtered_ndcg_values, marker='x', label='NDCG@5', linestyle='-')
plt.title("Evolution of AP and NCDG5 for each turn")
plt.xlabel("Turns")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(filtered_recall_values, filtered_precision_values, marker='o', label='Precision-Recall')
plt.title("Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

