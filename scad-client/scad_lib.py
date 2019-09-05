from operator import itemgetter
import numpy as np
from numpy import string_
import os, json, re
import sklearn.preprocessing as sk
#import matplotlib.pyplot as plt
#import seaborn as sb

# Compare two names according to 'matching_method' and return true or false. Supports only 'match:ATTRIBUTE' so far.
def matches(auth1, auth2, matching_method="", case_sensitive=False):
    r = False
    meth,att = matching_method.split(":")
    if meth == 'match':
        if not case_sensitive:
            if auth1[att].lower() == auth2[att].lower():
                r = True
        else:
            if auth1[att] == auth2[att]:
                r = True
    return r

# Compute binary p,r, and f values from dictionary of TP etc. counts
def bin_eval(rec):
    tp = rec['TP']
    fp = rec['FP']
    tn = rec['TN']
    fn = rec['FN']
    try:                        precision = float(tp / (tp + fp))
    except ZeroDivisionError:   precision = 1.0
    try:                        recall    = float(tp / (tp + fn)) 
    except ZeroDivisionError:   recall    = 0.0        
    try:                        f         = float((2*recall*precision)/(precision+recall))
    except ZeroDivisionError:   f         = 0.0    
    return {'p':precision, 'r':recall, 'f':f, 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'nid':rec['NID'], 'ncl':rec['NCL']} 

# Compare two authos ids and derive gold_label
def get_gold_label(aid_1, aid_2):
    label = ""
    # This happens when two author names match, but one or both are not annotated
    if aid_1 == "" or aid_2 == "" : label = "NID"
    elif aid_1 == aid_2           : label = "T"
    else                          : label = "F"
    return label

# Create descriptive string for each pub comparison, ane send to logger.
def log_printable_string(logger, pub_1, aid_1, pub_2, aid_2, result, name_attribute='shortname', only_show=[], suppress=[]):

    bin_class = result[0]['bin_class']
    if 'ALL' in suppress or bin_class in suppress or (len(only_show) > 0 and bin_class not in only_show):
        return 

    sys_label = result[0]['sys_label']
    gold_label = result[0]['gold_label']

    p="\n\t"+pub_1['id']+"\n\t'"+pub_1['title']+"'"
    a_temp=""
    for i,a in enumerate(pub_1['authors']):
        if i == aid_1:
            a_temp=a_temp+" *"+a[name_attribute]+"*; "
        else:
            a_temp=a_temp+" "+a[name_attribute]+"; "
    p=p+"\n\t"+a_temp.strip()[0:-1]

    p=p+"\n\t"+pub_2['id']+"\n\t'"+pub_2['title']+"'"
    a_temp=""
    for i,a in enumerate(pub_2['authors']):
        if i == aid_2:
            a_temp=a_temp+" *"+a[name_attribute]+"*; "
        else:
            a_temp=a_temp+" "+a[name_attribute]+"; "
    p=p+"\n\t"+a_temp.strip()[0:-1]

    if gold_label=="UNK":        
        bin_class = sys_label        

    p=p+"\n\t--------------------\n"
    for ev_name, exp_details, threshold, score, local_label, ev_details in result[1]:
        if ev_details == {}:
            ev_details = ""
        comp = " = "
        if   score >= threshold: comp=" >= "
        elif score < threshold:  comp=" < "
        p=p+"\t"+ev_name+": "+str(score)+comp+str(threshold)+": "+local_label+"\t\t"+ev_details+"\n"
    
    logger.info(p+"\t\t--> "+bin_class+"\n")

    return p+"\t\t--> "+bin_class+"\n"
        
   
