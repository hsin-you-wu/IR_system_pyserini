from pyserini.search.lucene import LuceneSearcher
import re
import json
from jnius import autoclass
import os
import pandas as pd
        
        
class JMSearcher(LuceneSearcher):
    """ Adds Jelinek-Mercer smoothing. """
    def __init__(self, index_dir: str):
        super().__init__(index_dir)
        
    def set_qld_jm(self, lambda_param: float):
        """ Configure the searcher to use Query Likelihood with Jelinek-Mercer smoothing. """  
        # import necessary classes
        LMJelinekMercerSimilarity = autoclass('org.apache.lucene.search.similarities.LMJelinekMercerSimilarity')
        IndexSearcher = autoclass('org.apache.lucene.search.IndexSearcher')
        similarity = LMJelinekMercerSimilarity(lambda_param)
        
        # re-initialize searcher
        self.object.searcher = IndexSearcher(self.object.reader)
        self.object.searcher.setSimilarity(similarity)

        
def make_queries(file_path, output_path):
    """
    - Given a raw query txt file, parse each query and write them to a jsonl file.
    """

    # read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # extract each raw query texts
    raw_queries = content.split('</top>')

    # parse each raw query and write to jsonl
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for raw_query in raw_queries:
            raw_query = raw_query.strip()
            if raw_query:
                query_num = re.search(r'<num> Number: (.+)', raw_query).group(1).strip()
                query_title = re.search(r'<title>(.+)', raw_query).group(1).strip()
                query_title = re.sub(r'[,.]', '', query_title) # remove comma and periods
                query = {query_num: query_title}
                jsonl_file.write(json.dumps(query) + '\n')     
    

def load_queries(jsonl_path):
    """
    Load queries from a JSONL file.
    """
    queries = []
    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            queries.append(json.loads(line))
    
    return queries


def rank_with_bm25(index_path:str, query_num:str, query_title:str, output_file_name:str) -> list:
    """ - Rank a set of indexes to a query with BM25. 
        - Store the results into a jsonl file.
    """
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=2, b=0.75) # set parameters
    hits = searcher.search(query_title, k=1000)

    # store results
    results = []
    for i in range(len(hits)):
        result = f'{query_num} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} bm25\n'
        if hits[i].score > 0:
            results.append(result)

    # write results
    output_file_path = f'results/{output_file_name}/bm25.txt'
    with open(output_file_path, 'a', encoding='utf-8') as file:
        for result in results:
            file.write(result)
    

def rank_with_jm(index_path:str, query_num:str, query_title:str, output_file_name:str) -> list:
    """ - Rank a set of indexes to a query with LM and Jelinek-Mercer smoothing. 
        - Store the results into a jsonl file.
    """
    searcher = JMSearcher(index_path)
    searcher.set_qld_jm(lambda_param=0.2)
    hits = searcher.search(query_title, k=1000)

    # store results
    results = []
    for i in range(len(hits)):
        result = f'{query_num} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} jm\n'
        if hits[i].score > 0:
            results.append(result)
    
    # write results
    output_file_path = f'results/{output_file_name}/jm.txt'
    with open(output_file_path, 'a', encoding='utf-8') as file:
        for result in results:
            file.write(result)


def rank_with_lp(index_path:str, query_num:str, query_title:str, output_file_name:str) -> list:
    """ - Rank a set of indexes to a query with LM and Laplace smoothing. 
        - Store the results into a jsonl file.
    """
    searcher = LuceneSearcher(index_path)

    # Calculate mu
    query_words = query_title.split(" ")
    total_terms =  len(query_words) 
    unique_terms = len(set(query_words))
    mu = total_terms / unique_terms

    searcher.set_qld()
    hits = searcher.search(query_title, k=1000)

    # store results
    results = []
    for i in range(len(hits)):
        result = f'{query_num} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} lp\n'
        if hits[i].score > 0:
            results.append(result)
    
    # write results
    output_file_path = f'results/{output_file_name}/lp.txt'
    with open(output_file_path, 'a', encoding='utf-8') as file:
        for result in results:
            file.write(result)


def rank(index_path:str, queries:list, output_file_name:str):
    """ 
    - Performs 3 ranking methods. 
    - Each ranking result will be stored in a jsonl file.
    """
    # rank with bm25 if not yet exists
    if not os.path.exists(f'results/{output_file_name}/bm25.txt'):
        os.makedirs(os.path.dirname(f'results/{output_file_name}/bm25.txt'), exist_ok=True)
        for i, query in enumerate(queries):
            query_num = list(query.keys())[0]
            query_title = list(query.values())[0]
            rank_with_bm25(index_path, query_num, query_title, output_file_name)

    # rank with lp if not yet exists
    if not os.path.exists(f'results/{output_file_name}/lp.txt'):
        os.makedirs(os.path.dirname(f'results/{output_file_name}/lp.txt'), exist_ok=True)
        for i, query in enumerate(queries):
            query_num = list(query.keys())[0]
            query_title = list(query.values())[0]
            rank_with_lp(index_path, query_num, query_title, output_file_name)

    # rank with jm if not yet exists
    if not os.path.exists(f'results/{output_file_name}/jm.txt'):
        os.makedirs(os.path.dirname(f'results/{output_file_name}/jm.txt'), exist_ok=True)
        for i, query in enumerate(queries):
            query_num = list(query.keys())[0]
            query_title = list(query.values())[0]
            rank_with_jm(index_path, query_num, query_title, output_file_name)


def results_to_df(result_folder_path:str, ans_file_path:str, output_csv_file:str):
    """ 
    - Create a training df from three result files.
    - Save the df to a csv file.
    """
    # parse ans file 
    rel_dict = {}
    with open(ans_file_path, 'r') as ans_file:
        for line in ans_file:
            elements = line.strip().split()
            ans_query_id = elements[0]
            ans_doc_id = elements[2]
            rel = int(elements[-1])
            rel_dict[(ans_query_id, ans_doc_id)] = rel

    # a dict to store accumulated data
    results = {}

    def parse_result_file(result_file_path: str, score_type: str):
        with open(result_file_path, 'r') as result_file:
            for line in result_file:
                elements = line.strip().split()
                query_id = elements[0]
                doc_id = elements[2]
                score = float(elements[4])

                key = (query_id, doc_id)
                if key not in results:
                    results[key] = {'query_id': query_id, 'doc_id': doc_id, 
                                    'score_bm25': None, 'score_lp': None, 'score_jm': None, 
                                    'rel': rel_dict.get(key, 0)}

                # assign the score to the correct column
                results[key][f'score_{score_type}'] = score

    # parse the three result files
    parse_result_file(os.path.join(result_folder_path, 'bm25.txt'), 'bm25')
    parse_result_file(os.path.join(result_folder_path, 'lp.txt'), 'lp')
    parse_result_file(os.path.join(result_folder_path, 'jm.txt'), 'jm')

    # build df and save to csv
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(output_csv_file, index=False)
    return df







