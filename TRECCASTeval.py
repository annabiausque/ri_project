from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import json
import numpy as np

from pandas import json_normalize
import pandas as pd

import rank_metric as metrics

class ConvSearchEvaluation:

    def __init__(self):

        # Training topics
        with open("./data/training/train_topics_v1.0.json", "rt", encoding="utf-8") as f: 
            self.train_topics = json.load(f)

        # fields: topic_turn_id, docid, rel
        self.train_relevance_judgments = pd.read_csv("./data/training/train_topics_mod.qrel", sep=' ', names=["topic_turn_id", "dummy", "docid", "rel"])

        # Test topics
        with open("./data/evaluation/evaluation_topics_v1.0.json", "rt", encoding="utf-8") as f: 
            self.test_topics = json.load(f)

        # fields: topic_turn_id, docid, rel
        self.test_relevance_judgments = pd.read_csv("./data/evaluation/evaluation_topics_mod.qrel", sep=' ', names=["topic_turn_id", "dummy", "docid", "rel"])

        set_of_conversations = set(self.train_relevance_judgments['topic_turn_id'])
        self.judged_conversations = np.unique([a.split('_', 1)[0] for a in set_of_conversations])

    def num_rel(self, topic_turn_id):
                # Try to get the relevance judgments from the TRAINING set
        aux = self.train_relevance_judgments.loc[self.train_relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        # IF fail, try to get the relevance judgments from the TEST set
        if np.size(aux) == 0 :
            aux = self.test_relevance_judgments.loc[self.test_relevance_judgments['topic_turn_id'] == (topic_turn_id)]
        
        rel_docs = aux.loc[aux['rel'] != 0]
        
        return rel_docs.count().rel
        
        
    def eval(self, result, topic_turn_id):
        total_retrieved_docs = result.count()._id

        # Try to get the relevance judgments from the TRAINING set
        aux = self.train_relevance_judgments.loc[self.train_relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        # IF fail, try to get the relevance judgments from the TEST set
        if np.size(aux) == 0 :
            aux = self.test_relevance_judgments.loc[self.test_relevance_judgments['topic_turn_id'] == (topic_turn_id)]
        
        rel_docs = aux.loc[aux['rel'] != 0]
        
        query_rel_docs = rel_docs['docid']
        relv_judg_list = rel_docs['rel']
        total_relevant = relv_judg_list.count()

        # P@10
        top10 = result['_id'][:10]
        true_pos= np.intersect1d(top10,query_rel_docs)
        p10 = np.size(true_pos) / 10
        
        true_pos= np.intersect1d(result['_id'],query_rel_docs)
        recall = np.size(true_pos) / total_relevant

        # Compute vector of results with corresponding relevance level 
        relev_judg_results = np.zeros((total_retrieved_docs,1))
        for index, doc in rel_docs.iterrows():
            relev_judg_results = relev_judg_results + ((result['_id'] == doc.docid)*doc.rel).to_numpy()
        
        # Normalized Discount Cummulative Gain
        p10 = metrics.precision_at_k(relev_judg_results[0], 10)
        ndcg5 = metrics.ndcg_at_k(r = relev_judg_results[0], k = 5, method = 1)
        ap = metrics.average_precision(relev_judg_results[0], total_relevant)
        mrr = metrics.mean_reciprocal_rank(relev_judg_results[0])
        
        if False:
            print("Prec@10: ", p10)
            print("NDCG@5: ", ndcg5)
            print("AP: ", ap)
            print("MRR: ", mrr)
        
        return [p10, recall, ap, ndcg5]
    
    
    def evalPR(self, scores, topic_turn_id):
    
        aux = self.train_relevance_judgments.loc[self.train_relevance_judgments['topic_turn_id'] == (topic_turn_id)]
        
        # IF fail, try to get the relevance judgments from the TEST set
        if np.size(aux) == 0 :
            aux = self.test_relevance_judgments.loc[self.test_relevance_judgments['topic_turn_id'] == (topic_turn_id)]

        idx_rel_docs = aux.loc[aux['rel'] != (0)]

        [dummyA, rank_rel, dummyB] = np.intersect1d(scores['_id'],idx_rel_docs['docid'], return_indices=True)
        rank_rel = np.sort(rank_rel) + 1
        total_relv_ret = rank_rel.shape[0]
        if total_relv_ret == 0:
            return [np.zeros(11,), [], total_relv_ret]

        recall = np.arange(1, total_relv_ret+1)/idx_rel_docs.shape[0]
        precision = np.arange(1, total_relv_ret+1)/rank_rel

        precision_interpolated = np.maximum.accumulate(np.flip(precision))
        recall_11point = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        precision_11point = np.interp(recall_11point, recall, np.flip(precision_interpolated))

        if False:
            print(total_relv_ret)
            print(rank_rel)
            print(recall)
            print(precision)
            plt.plot(recall, precision, color='b', alpha=1) # Raw precision-recall
            plt.plot(recall, precision_interpolated, color='r', alpha=1) # Interpolated precision-recall
            plt.plot(recall_11point, precision_11point, color='g', alpha=1) # 11-point interpolated precision-recall

        return [precision_11point, recall_11point, total_relv_ret]
