import time, json, re, requests, logging, sys, os, itertools
from collections import Counter
from scad_lib import matches, get_gold_label, log_printable_string, bin_eval,  evaluate_conll
from tqdm import tqdm

class simple_scad_client(object):

    pub_list          = []
    data_type         = 'UNKNOWN'
    skip_duplicates   = True
    verbose           = False
    logger            = None
    scad_server_url   = ""
    params            = {}
    blocking_pattern  = ""

    def __init__(self, scad_server_url="", logfile="", verbose=False):

        if logfile == "":
            logfilename = "./log/scad_log_"+str(os.getpid())+".txt"
        else:
            logfilename=logfile
        os.makedirs(os.path.dirname(logfilename), exist_ok=True)

        # Default: Do not overwrite log
        logging.basicConfig(filename=logfilename, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()
        self.verbose = verbose
        if verbose:
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
        self.scad_server_url = scad_server_url
        self.logger.info("Created simple_scad_client for communication with server at %s, logging to %s"%(scad_server_url, logfilename))


    def load_publications(self, pub_file_name="", blocking_pattern=""):
        self.logger.info("Loading data from '%s'"%pub_file_name)
        pat = ".*"+blocking_pattern+".*"
        block_hits = Counter()        
        if blocking_pattern != "":
            self.logger.info("Setting blocking_pattern to .*%s.*"%blocking_pattern)
            self.blocking_pattern = blocking_pattern       
        else:
            self.logger.info("WARNING: No blocking pattern provided! Processing will take *LONG* for any non-trivial input file size. Continue at your own risk!")
            # Make sure this appears on screen regardless of loglevel
            print("WARNING: No blocking pattern provided! Processing will take *LONG* for any non-trivial input file size. Continue at your own risk!")
            time.sleep(5)

        with open(pub_file_name) as in_file:
            for l in json.load(in_file)['publications']:
                if blocking_pattern != "":
                    m = re.match(pat, str(l), re.IGNORECASE)    # Apply blocking_pattern as early as possible, i.e. at loading time
                    if m != None:
                        try:
                            block_hits.update({m.groups()[0]:1})
                        except IndexError:
                            pass
                            # No group in RE, block match statistics will not work
                        self.pub_list.append(l)
                else:
                    self.pub_list.append(l)                         # No blocking_pattern

        if len(self.pub_list)>0:
            self.data_type = self.pub_list[0]['id'].split(":")[0]
        self.logger.info("Publication data type is '%s'"%self.data_type)

        if blocking_pattern != "":
            self.logger.info("Block match statistics:")
            for (n,p) in block_hits.most_common():
                self.logger.info("\t%s\t(%s %%)\t%s"%(str(p), '{:{width}.{prec}f}'.format(p*100/len(self.pub_list), width=6, prec=2), str(n)))
      
        
    def load_resources(self, resourcefile):
        self.logger.info("Loading server-side resources from '%s'"%resourcefile)
        with open(resourcefile) as jin:
            requests.post(self.scad_server_url+"/init_scad_resources", json = json.load(jin))



    def load_params(self, paramfile):
        self.logger.info("Loading match params from '%s'"%paramfile)
        with open(paramfile) as jin:        
            self.params = json.load(jin)
        self.logger.info("Setting data_type param to previously detected '%s'"%self.data_type)
        self.params['data_type']=self.data_type
        
        
    def match_publications(self, evaluate=False, name_matching_method=""):
        pub_comparisons, record_comparisons, skipped  = 0, 0, 0        
        pair_results = {'TP':0, 'FP':0, 'FN':0, 'TN':0, 'NID':0, 'NCL':0, 'UNK':0}      
          
        if not self.verbose: 
            bar = tqdm(total=int((len(self.pub_list)*(len(self.pub_list)-1))/2), ncols=80)
            
        pat = ".*"+self.blocking_pattern+".*"
        
        matches_as_edges=[]
        gold_clusters={}
        gold_ids=[]
        for e, pub_1 in enumerate(self.pub_list):
            for pub_2 in self.pub_list[e+1:]:
                if not self.verbose: bar.update(1)
                if pub_1['id'] == pub_2['id']: continue # These do not count as skipped

                if self.skip_duplicates:
                    if pub_1['title'] == pub_2['title']:
                        skipped+=1
                        self.logger.info("Skipping pubs with identical title but different ids")
                        continue
                    if self.data_type == "dblp" and pub_1['id'].split("/")[2] == pub_2['id'].split("/")[2]:
                        # dblp:journals/informs/SheraliSL00
                        skipped+=1
                        self.logger.info("Skipping identical pubs in different venues (DBLP)")
                        continue               

                pub_comparisons+=1
                for tup in itertools.product( [(e,p) for e,p in enumerate(pub_1['authors']) if re.match(pat, str(p), re.IGNORECASE) != None], 
                                              [(e,p) for e,p in enumerate(pub_2['authors']) if re.match(pat, str(p), re.IGNORECASE) != None]):
                    auth1=tup[0][1]
                    auth2=tup[1][1]
                    
                    if matches(auth1, auth2, matching_method=name_matching_method, case_sensitive=False):
                        # If auth1 is identified, get its cluster no and add to gold_clusters, else do not add to gold_clusters
                        if auth1['id'] != "":
                            if auth1['id'] not in gold_ids:
                                gold_ids.append(auth1['id'])                         
                            gold_clusters[pub_1['id']+"@"+str(tup[0][0])]=gold_ids.index(auth1['id'])

                        # If auth2 is identified, get its cluster no and add to gold_clusters, else do not add to gold_clusters
                        if auth2['id'] != "":
                            if auth2['id'] not in gold_ids:
                                gold_ids.append(auth2['id'])                         
                            gold_clusters[pub_2['id']+"@"+str(tup[1][0])]=gold_ids.index(auth2['id'])

                        # gold_clusters contains all identified records, including singletons 
                        record_comparisons+=1
                        if evaluate : self.params['gold_label'] = get_gold_label(auth1['id'], auth2['id'])
                        else        : self.params['gold_label'] = 'UNK'

                        result = json.loads(requests.post(self.scad_server_url+'/scad_api', json = {'pub_1':pub_1, 'ai_1':tup[0][0], 'pub_2':pub_2, 'ai_2':tup[1][0], 'params':self.params}).text)
                        
                        # Create result edge for later clustering.
                        # Consider only pairs that *can* actually be evaluated.
                        if result[0]['sys_label'] == "T" and auth1['id'] != "" and auth2['id'] != "":
                            matches_as_edges.append((pub_1['id']+"@"+str(tup[0][0]),pub_2['id']+"@"+str(tup[1][0])))

                        # Update counter for TP, FP etc.
                        pair_results[result[0]['bin_class']]+=1
                        log_printable_string(self.logger, pub_1, tup[0][0], pub_2, tup[1][0], result, name_attribute='shortname', only_show=[], suppress=['NCL','NID'])
        if not self.verbose:  
            bar.close()

        bc_p,bc_r,bc_f = evaluate_conll(matches_as_edges, gold_clusters, "test", "./reference-coreference-scorers/scorer.pl")
#        print("B-CUB P: %s R: %s F: %s"%(p,r,f))
        bin_result = bin_eval(pair_results)
        ms = "Used methods:\n"
        for m in self.params['methods']:
            if 'method_name' in m:
                ms=ms+json.dumps(m, indent=1)+"\n"
        self.logger.info("\n################\n"+ms+"\n%s\tpublications in block %s yielded %s PUBLICATION and %s AMBIGUOUS RECORD comparisons."%(str(len(self.pub_list)), pat, str(pub_comparisons), str(record_comparisons)))
        self.logger.info("%s\tpublication pairs were skipped (duplicates)!"%(str(skipped)))
        self.logger.info("%s\tambiguous record pairs were not classified (lack of evidence)!"%(str(bin_result['ncl'])))
        if evaluate:
            self.logger.info("\n\t*B-CUBED* clustering evaluation\n\tP:\t%s\n\tR:\t%s\n\tF:\t%s\n\t"%(bc_p, bc_r, bc_f))
            self.logger.info("\n\t*Binary* classification evaluation\n\tP:\t%s\n\tR:\t%s\n\tF:\t%s\n\t(TP: %s, FP: %s, TN: %s, FN: %s)"%(bin_result['p'], bin_result['r'], bin_result['f'], bin_result['tp'],bin_result['fp'],bin_result['tn'],bin_result['fn']))
            self.logger.info("\t%s ambiguous record pairs were ignored for evaluation because at least one author had no gold id!"%(str(bin_result['nid'])))

        if not self.verbose:
            print("\n################\n"+ms+"\n%s\tpublications in block %s yielded %s PUBLICATION and %s AMBIGUOUS RECORD comparisons."%(str(len(self.pub_list)), pat, str(pub_comparisons), str(record_comparisons)))
            print("%s\tpublication pairs were skipped (duplicates)!"%(str(skipped)))
            print("%s\tambiguous record pairs were not classified (lack of evidence)!"%(str(bin_result['ncl'])))
            if evaluate:
                print("\n\t*B-CUBED* clustering evaluation\n\tP:\t%s\n\tR:\t%s\n\tF:\t%s"%(bc_p, bc_r, bc_f))
                print("\n\t*Binary* classification evaluation")
                print("\tP:\t%s\n\tR:\t%s\n\tF:\t%s\n\t(TP: %s, FP: %s, TN: %s, FN: %s)"%(bin_result['p'], bin_result['r'], bin_result['f'], bin_result['tp'],bin_result['fp'],bin_result['tn'],bin_result['fn']))
                print("\t%s ambiguous record pairs were ignored for evaluation because at least one author had no gold id!"%(str(bin_result['nid'])))
       
