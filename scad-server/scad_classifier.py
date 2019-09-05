from wombat_api.core import connector as wb_conn
from sem_sim_lib import get_vector_tuples, tuple_average, WOMBAT_VEX, FASTTEXT_VEX, ELMO_VEX
from scad_methods import scad
from threading import Lock

class scad_classifier(object):

    # Heavy-weight resources are kept on the server side, i.e. here
    # This will be populated by call to 
    CACHE                = None
     
    def __init__(self):
        print("Creating classifier from %s"%str(type(self)))
        if self.CACHE == None:
            print("Creating global CACHE")
            self.CACHE = {}

    # This returns sys_label UNK if no evidence was found, else F or T based on CONNECTOR-type combined result of all methods
    def match_authors(self, pub_1, ai_1, pub_2, ai_2, params):

        collected_results = []
        global_language = params.pop('global_lang')
        data_type       = params.pop('data_type')

        connector=params['connector'].lower()   # OR or AND or MAJ
        local_matches=[]                # Collect T F U results from each method call, interpret using connector above

        # Iterate over all methods to be applied, plus their params
        for method_params in params['methods']:
            try:
                method = method_params.pop('method_name')
            except KeyError:
                continue
            threshold = float(method_params.pop('threshold'))
            # *Add* some method-invariant params
            method_params.update({'cache':self.CACHE, 'pub_1':pub_1, 'ai_1':ai_1, 'pub_2':pub_2, 'ai_2':ai_2, 'lang':global_language, 'data_type':data_type})

            # results is a dict from experiment keys to dicts containing 'score' and 'evidence'
            # A score of -1 means method could not decide (e.g. lack of coauthors)
            results = getattr(scad,'scad_'+method)(**method_params)
            # Extract individual results (will be only one in most cases)
            for key in results:
                score = float(results[key]['score'])
                if score == -1:
                    local_matches.append('U')
                elif score >= threshold:
                    local_matches.append('T')
                else:
                    local_matches.append('F')
                collected_results.append((method, key, threshold, score, local_matches[-1], results[key]['evidence']))
       
        final_match='U'
        match_set=set(local_matches)
        if connector=='and':
            # *All* filters must match, U is not allowed here
            if 'F' not in match_set and 'U' not in match_set:
                final_match='T'
            else:
                final_match='F'
        elif connector == 'or':
            if 'T' in match_set:
                final_match='T'
            else:
                final_match='F'
        # add MAJ
        
        gold_label = params['gold_label']
        if   gold_label == 'T' and final_match == 'T'     : bin_class='TP'                           
        elif gold_label == 'F' and final_match == 'F'     : bin_class='TN'            
        elif gold_label == 'T' and final_match == 'F'     : bin_class='FN'            
        elif gold_label == 'F' and final_match == 'T'     : bin_class='FP'
        elif final_match == 'U'                           : bin_class='NCL'            
        elif gold_label == 'NID' or final_match == 'NID'  : bin_class='NID'
        elif gold_label == 'UNK'                          : bin_class='UNK'

        return {'bin_class':bin_class, 'sys_label':final_match, 'gold_label':params['gold_label']}, collected_results                        

        
