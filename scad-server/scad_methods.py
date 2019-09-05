from sem_sim_lib import preprocess, get_vector_tuples, WOMBAT_VEX, tuple_average, get_top_n_cos_sim_avg
import pickle, numpy as np, scipy.spatial.distance as dist
from threading import Lock
class scad:

    def scad_match_all(pub_1=None, ai_1=None, pub_2=None, ai_2=None, cache=None,  previous_evidence=None, 
                        fold=False, unit='token', lang='', data_type='', **ignore):
        return {'match_all':{'score':1.0, 'evidence':{}}}



    def scad_unit_overlap(pub_1=None, ai_1=None, pub_2=None, ai_2=None, cache=None,  previous_evidence=None, 
                          fold=False, unit='token', tf=False, idf=False, lang='', data_type='', **ignore):
        local_result = {}
        mname="unit_overlap"
        experiment_key_base=mname+'#fold:'+str(fold)+'#unit'+unit
        (tfdict1, units1) = preprocess(pub_1['title'], cache, fold=fold, unit=unit, lang=lang)
        (tfdict2, units2) = preprocess(pub_2['title'], cache, fold=fold, unit=unit, lang=lang)

        evidence=set(units1).intersection(set(units2))
        score = len(evidence)
        return {experiment_key_base:{'score':score, 'evidence':str(evidence) if len(evidence)>0 else "" }}


    # WOMBAT only
    def scad_cos_of_avg(pub_1=None, ai_1=None, pub_2=None, ai_2=None, cache=None,  previous_evidence=None, 
                        fold=False, unit='token', emb=None, tf=False, idf=False, lang='', data_type='', **ignore):
        # This contains the results of all experiments; might be more than one if more than one emb has been supplied
        local_result = {}
        mname="cos_of_avg"
        experiment_key_base=mname+'#fold:'+str(fold)+'#unit'+unit
        try:
            # This method caches tuple reps per pub, tuple averages, and tf counts
            key = experiment_key_base + pub_1['id'] + emb + str(tf) + str(idf)
            (tups1, tfdict1, avg1) = cache[key]
        except KeyError:
            (tfdict1, units1) = preprocess(pub_1['title'], cache, fold=fold, unit=unit, lang=lang)
            tups1 = get_vector_tuples([units1], WOMBAT_VEX, {'wombat':cache['wombat'], 'emb':emb})        
            avg1  = tuple_average(tups1, idfdict = cache[unit+'_'+data_type+'_idf'] if idf else {}, tfdict = tfdict1 if tf else {})
            cache[key] = (tups1, tfdict1, avg1)
        try:
            key = experiment_key_base+pub_2['id'] + emb + str(tf) + str(idf)
            (tups2, tfdict2, avg2) = cache[key]
        except KeyError:
            (tfdict2, units2) = preprocess(pub_2['title'], cache, fold=fold, unit=unit, lang=lang)
            tups2 = get_vector_tuples([units2], WOMBAT_VEX, {'wombat':cache['wombat'], 'emb':emb})        
            avg2  = tuple_average(tups2, idfdict = cache[unit+'_'+data_type+'_idf'] if idf else {}, tfdict = tfdict2 if tf else {})
            cache[key]=(tups2, tfdict2, avg2)
        sem_sim  = 1 - dist.cosine(avg1, avg2)
        # experimental params are encoded in key
        experiment_key = experiment_key_base+'#'+emb+'#tf:'+str(tf)+'#idf:'+str(idf)
        local_result[experiment_key] = {'score':sem_sim, 'evidence': {}}
        return local_result
        

    def scad_avg_of_cos(pub_1=None, ai_1=None, pub_2=None, ai_2=None, cache=None, previous_evidence=None,
                        fold=False, unit='token', emb=None, tf=False, idf=False, top_n=-1, lang='', data_type='', **ignore):
        local_result = {}
        mname="avg_of_cos"
        # Invariant key
        experiment_key_base=mname+'#fold:'+str(fold)+'#unit'+unit
        # preprocess is cached
        (tfdict1, units1) = preprocess(pub_1['title'], cache, fold=fold, unit=unit, lang=lang)
        key = 'tuples'+unit+emb+str(fold)+str(units1)
        try:
            tups1 = cache[key]
        except KeyError:
            tups1 = get_vector_tuples([units1], WOMBAT_VEX, {'wombat':cache['wombat'], 'emb':emb})        
            cache[key] = tups1

        (tfdict2, units2) = preprocess(pub_2['title'], cache, fold=fold, unit=unit, lang=lang)
        
        key = 'tuples'+unit+emb+str(fold)+str(units2)
        try:
            tups2=cache[key]
        except KeyError:
            tups2 = get_vector_tuples([units2], WOMBAT_VEX, {'wombat':cache['wombat'], 'emb':emb})
            cache[key] = tups2

        r = get_top_n_cos_sim_avg(tups1, tfdict1, tups2, tfdict2, top_n, tf, idf, cache[unit+'_'+data_type+'_idf'])

        experiment_key = experiment_key_base+'#'+emb+'#tf:'+str(tf)+'#idf:'+str(idf)+'#top_n'+str(top_n)
        local_result[experiment_key] = {'score':r[0], 'evidence':str(r[1])}
        return local_result


    # TODO: weight locally by name frequency / likelihood
    def scad_local_coauthor_similarity(pub_1=None, ai_1=None, pub_2=None, ai_2=None, cache=None, previous_evidence=None,
                                       scheme='shortname', match='binary', measure='overlap', lang=None, data_type='', **ignore):
        mname="loc_coauth_sim"
        # Invariant key
        experiment_key_base=mname+'#scheme:'+scheme+'#match:'+match+'#measure:'+measure
        
        coauthors1=set([a[scheme] for i,a in enumerate(pub_1['authors']) if i != ai_1])
        coauthors2=set([a[scheme] for i,a in enumerate(pub_2['authors']) if i != ai_2])
        len1 = len(coauthors1)
        len2 = len(coauthors2)

        score = -1.0
        evidence = {}
        # If at least one is single author, we have no ca-evidence whatsoever
        if len1 != 0 and len2 != 0:
            if match == 'binary':
                matching       = set(coauthors1.intersection(coauthors2))
                if len(matching)>0:
                    evidence       = mname+':'+str(matching)
                n_matching = len(matching)
                if measure == 'overlap':
                    score = n_matching
                elif measure == 'ratio':
                    score = n_matching / ((len1+len2)-n_matching)
                else:
                    pass
                    
        return {experiment_key_base:{'score':score, 'evidence':evidence}}

