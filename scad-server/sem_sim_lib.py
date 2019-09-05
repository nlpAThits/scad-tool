from wombat_api.core import connector as wb_conn
import numpy as np, scipy.spatial.distance as dist
from operator import itemgetter
from collections import Counter

WOMBAT_VEX   = 1
FASTTEXT_VEX = 2
ELMO_VEX     = 3


def tokenize(line, sw_symbol="", conflate=False, FWORDS={}, fold=False, pretokenizer=None):

    tokens=[]
    for t in pretokenizer.tokenize(line, agressive_dash_splits=False, escape=False):
        # Get each raw token
        t = t.strip()
        if fold: t=t.lower()
        # With each token
        while(True):            
            modified=False
            if t.isnumeric(): 
                t="0"*len(t)
                break   # No more mods apply, break here

            if len(t)==1 and is_plain_alpha(t)==False:
                t=""                    
                break   # No more mods apply, break here                

            if t.lower() in FWORDS and sw_symbol!=None:
                # t is a stopword, which is to be treated somehow
                if conflate:
                    # stopword conflation is active
                    try:
                        # If the last token is a sw already, skip current sw token
                        if tokens[-1]==sw_symbol:
                            t=""
                            break
                        else:
                            t=sw_symbol
                            break
                    except IndexError:
                        # current sw is first token
                        t=sw_symbol
                        break
                else:
                    t=sw_symbol # sw-symbol might also be empty, in which case an empty string is added here, which is removed later
                    break   # No more mods apply, break here

            if t!="" and t not in {'2d', '2D', '3d', '3D', "'s"}:
                if is_plain_alpha(t[0])==False:
                    t=t[1:].strip()
                    modified=True
                elif is_plain_alpha(t[-1])==False:
                    t=t[:-1].strip()
                    modified=True                        
            if modified==False: # Do this as last point in check
                break
        if t!="": tokens.append(t)
    return tokens


def is_plain_alpha(t):
    o=ord(t)
    if  (o>= 65 and  o<=90) or\
        (o>= 97 and o<=122) or\
        (o>=192 and o<=214) or\
        (o>=216 and o<=246) or\
        (o>=248 and o<=476) or\
        (o>=512 and o<=591):
        return True
    else:
        return False


def weight_tuples(tuplelist, tfdict={}, idfdict={}):
    wtuples=[]
    for t in tuplelist:
        try:                tf = tfdict[t[0]]
        except KeyError:    tf = 1.0
        try:                idf = idfdict[t[0]]
        except KeyError:    idf = 1.0
        wtuples.append((t[0],t[1],tf*idf))
    wtuples.sort(key = itemgetter(2), reverse = True)
    return wtuples


def get_vector_tuples(types, vex_source, params):
    if vex_source == WOMBAT_VEX:
        wb=params['wombat']
        emb=params['emb']
        # raw = False because tokens have been preprocessed already
        tuples=wb.get_vectors(emb, {}, for_input=types, raw=False, in_order=False, ignore_oov=True, as_tuple=True, default=0.0)[0][1][0][2]
    return tuples

def tuple_average(tuplelist, tfdict={}, idfdict={}):
    vecs=[]
    for t in tuplelist:
        try:
            tfw=tfdict[t[0]]
        except KeyError:
            tfw=1
        try:
            idfw=idfdict[t[0]]
        except KeyError:
            idfw=1.0
        for u in range(tfw):
            vecs.append(t[1]*idfw)
    return np.nanmean(vecs, axis=0)    


def preprocess(text, cache, fold=False, unit='token', lang=''):
    local_lang=lang # Mod to make language-dependent

    key = 'prepro'+local_lang+unit+str(fold)+text
    try:
        return cache[key]
    except KeyError:
        tokens = tokenize(text, FWORDS=cache[local_lang+'_stopwords'], fold=fold, pretokenizer=cache[local_lang+'_pretokenizer'])
        if unit == 'stem':
            stems=[]
            for t in tokens:
                stems.append(cache[local_lang+'_stemmer'].stem(t))
            tokens = stems
        cache[key] = (Counter(tokens), tokens)
    return cache[key]
    


def get_top_n_cos_sim_avg(tuples1, tfdict1, tuples2, tfdict2, top_n, tf_weighting, idf_weighting, TOKEN_IDF, sort_evidence=False, yield_matrix=False):

    # Convert tuples to weightedtuples of (word, vec, weight)
    weightedtuples1 = weight_tuples(tuples1, 
        tfdict=tfdict1 if tf_weighting else {}, 
        idfdict=TOKEN_IDF if idf_weighting else {})

    weightedtuples2 = weight_tuples(tuples2, 
        tfdict=tfdict2 if tf_weighting else {}, 
        idfdict=TOKEN_IDF if idf_weighting else {})

    n_weightedtuples1,n_weightedtuples2=[],[] 
    n1,n2=0,0
    o_max,i_max=50,50

    # If no weighting is applied, there will only be *one* rank group with weight 1.0.
    # That means all pairs will be considered
    # Determine no of lines covered by top n distinct rank groups
    # Go over all tuples in weighting order and find cut-off point
    for n1 in range(len(weightedtuples1)):
        weight=weightedtuples1[n1][2]   # Get weight of current tuple
        if weight in n_weightedtuples1:
            continue # weight has been seen, so current tuple belongs in current rank group
        elif len(n_weightedtuples1)<top_n:
            n_weightedtuples1.append(weight) # we have not yet seen all n different weights
        else:
            break

    xwords=None    
    if yield_matrix:
        # n1 is the cut-off point in weightedtuples1
        xwords = [(word+" "+'{0:.3f}'.format(weight), weight in n_weightedtuples1) for (word,_,weight) in weightedtuples1[0:o_max]]
    
    # Repeat for 2
    for n2 in range(len(weightedtuples2)):
        weight=weightedtuples2[n2][2]   # Get weight of current tuple
        if weight in n_weightedtuples2:
            continue # weight has been seen
        elif len(n_weightedtuples2)<top_n:
            n_weightedtuples2.append(weight) # we have not yet seen all n different weights
        else:
            break
    ywords = None
    if yield_matrix:
        # n2 is the cut-off point in weightedtuples2
        ywords = [(word+" "+'{0:.3f}'.format(weight), weight in n_weightedtuples2) for (word,_,weight) in weightedtuples2[0:i_max]]


    matrix = None
    matches = None
    if yield_matrix:
        matrix=np.zeros((i_max,o_max))            

        for o_i in range(len(xwords)):      # Iterate over all tuples in wtups1 up to 2*cut-off point --> X
            o_tup = weightedtuples1[o_i]
            for i_i in range(len(ywords)):  # Iterate over all tuples in wtups2 up to 2*cut-off point --> Y
                i_tup = weightedtuples2[i_i]
                matrix[i_i][o_i]=1-dist.cosine(o_tup[1],i_tup[1])

    # Pad tuple lists
    weightedtuples1=weightedtuples1+([('dummy',0,0)]*n1)
    weightedtuples2=weightedtuples2+([('dummy',0,0)]*n2)

    all_sim_pairs=[]
    for o_i,o_tup in enumerate(weightedtuples1[0:n1+1]):            # Iterate over all tuples in wtups1 up to cut-off point+1 --> X
        for i_i, i_tup in enumerate(weightedtuples2[0:n2+1]):       # Iterate over all tuples in wtups2 up to cut-off point+1 --> Y
            cs=1-dist.cosine(o_tup[1],i_tup[1])                     # Compute pairwise sim as 1 - dist
            all_sim_pairs.append((o_tup[0]+" & "+i_tup[0],cs))      # Collect all sim pairs

    #####################################################        
    # Option:                                           # 
    # Use only top n pairs for averaging                #
    # Sort by pairwise sim (high to low)                #
    all_sim_pairs.sort(key=itemgetter(1), reverse=True) #
    if top_n > 1:                                       #
        all_sim_pairs=all_sim_pairs[:top_n]             # 
    #####################################################

    matches = None
    if yield_matrix:
        matches=[(s.split(" & ")[0], s.split(" & ")[1]) for (s,_) in all_sim_pairs]

    # Collect and average over all pairwise sims
    sim=np.average([c for (_,c) in all_sim_pairs])
    if sort_evidence:
        # Sort alphabetically by sim pair (for output purposes only)
        all_sim_pairs.sort(key=itemgetter(0))

    return (sim, all_sim_pairs, (xwords, ywords, matrix, matches))

