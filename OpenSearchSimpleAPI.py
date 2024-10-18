from opensearchpy import OpenSearch
from opensearchpy import helpers
from opensearchpy.exceptions import TransportError
import json
import pprint as pp

from pandas import json_normalize
import pandas as pd
import rank_metric

class OSsimpleAPI:

    def __init__(self):
        host = 'api.novasearch.org'
        port = 443

        user = 'kwizard' # Add your user name here.
        password = 'oFVS/[6Q3b3>' # Add your user password here. For testing only. Don't store credentials in code. 

        self.index_name = 'kwiz'
        self.client = OpenSearch(
            hosts = [{'host': host, 'port': port}],
            http_compress = True, # enables gzip compression for request bodies
            http_auth = (user, password),
            use_ssl = True,
            url_prefix = 'opensearch',
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False
        )

        self.client.indices.open(self.index_name)
        
        if self.client.indices.exists(self.index_name):
            resp = self.client.indices.open(index = self.index_name)
            print(resp)

            print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
            settings = self.client.indices.get_settings(index = self.index_name)
            pp.pprint(settings)

            print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
            mappings = self.client.indices.get_mapping(index = self.index_name)
            pp.pprint(mappings)

            print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
            print(self.client.count(index = self.index_name))
        else:
            print("Index does not exist.")

    def search_body(self, query=None, numDocs=10):
        search_query = {
            "query": {
                "match": {
                    "contents": {
                        'query': query
                    }
                }
            }
        }
    
        result = self.client.search(index=self.index_name, body=search_query, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def search_entities(self, query=None, numDocs=10):
        result = self.client.search(index=self.index_name, body={"query": {"entities": {"contents": {'query': query}}}}, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def search_QSL(self, query_qsl=None, numDocs=10):
        result = self.client.search(index=self.index_name, body=query_qsl, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def termvectors_JSON(self, doc_id):
        term_statistics_json = self.client.termvectors(self.index_name, id=doc_id, fields='contents', field_statistics="true", term_statistics ='true')
        return term_statistics_json

    def analyzer(self, query, analyzer):
        tokens = self.client.indices.analyze(index=self.index_name, body = {"analyzer": analyzer, "text": query})
        return tokens
    
    def query_terms(self, query, analyzer):
        tokens = self.client.indices.analyze(index=self.index_name, body = {"analyzer": analyzer, "text": query})
        norm_query = ""
        for term in tokens['tokens']: 
            norm_query = norm_query + " " + term['token']

        return norm_query
    
    def get_doc_body(self, doc_id):
        aa = self.client.get(self.index_name, id=doc_id)["_source"]["contents"]
        return aa

    
    ####################################################################################
    #### Some JSON cleaning
    
    def doc_term_vectors(self, doc_id):
        term_statistics_json = self.client.termvectors(self.index_name, id=doc_id, fields='contents', field_statistics="true", term_statistics ='true')
        doc_freq = term_statistics_json["term_vectors"]["contents"]["field_statistics"]["doc_count"]
        sum_doc_freq = term_statistics_json["term_vectors"]["contents"]["field_statistics"]["sum_doc_freq"]
        sum_ttf = term_statistics_json["term_vectors"]["contents"]["field_statistics"]["sum_ttf"]
        term_statistics={}
        for term in term_statistics_json["term_vectors"]["contents"]["terms"]:
            term_statistics[term] = [term_statistics_json["term_vectors"]["contents"]["terms"][term]["term_freq"], term_statistics_json["term_vectors"]["contents"]["terms"][term]["doc_freq"], term_statistics_json["term_vectors"]["contents"]["terms"][term]["ttf"]]
        return doc_freq, sum_doc_freq, sum_ttf, term_statistics

    def multi_doc_term_vectors(self, doc_ids):
        term_statistics_json = self.client.mtermvectors(self.index_name, ids=doc_ids, fields='contents', field_statistics="true", term_statistics ='true')
        docs_term_vectors={}

        doc_freq = term_statistics_json["docs"][0]["term_vectors"]["contents"]["field_statistics"]["doc_count"]
        sum_doc_freq = term_statistics_json["docs"][0]["term_vectors"]["contents"]["field_statistics"]["sum_doc_freq"]
        sum_ttf = term_statistics_json["docs"][0]["term_vectors"]["contents"]["field_statistics"]["sum_ttf"]

        for doc in term_statistics_json["docs"]:
            term_statistics={}
            doc_id = doc["_id"]
            for term in doc["term_vectors"]["contents"]["terms"]:
                term_statistics[term] = [doc["term_vectors"]["contents"]["terms"][term]["term_freq"], doc["term_vectors"]["contents"]["terms"][term]["doc_freq"], doc["term_vectors"]["contents"]["terms"][term]["ttf"]]
            docs_term_vectors[doc_id] = term_statistics

        return doc_freq, sum_doc_freq, sum_ttf, docs_term_vectors
    
    ####################################################################################
