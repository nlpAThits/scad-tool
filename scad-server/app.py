import json, sys, logging
from scad_classifier import scad_classifier
from flask import Flask, request, jsonify, make_response, render_template, session
from exceptions import BadRequest
from threading import Lock
from wombat_api.core import connector as wb_conn

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.moses import MosesTokenizer

app = Flask(__name__)

# Silence werkzeug logger
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app.secret_key = "scad"
# Create *one* classifier instance to be used by accesses from *all* threads
classifier = scad_classifier()

@app.route('/scad_api', methods=['POST'])
def scad_api():
    datastring = request.data.decode().strip()
    try:
        data = json.loads(datastring)
    except ValueError:
        error = BadRequest('Invalid JSON given in request: {data}'.format(data=datastring))
        LOG.info('BadRequest received with following data: {data}'.format(data=request.data))        
        return make_response(jsonify(error.to_dict()), error.status_code)

    return jsonify(classifier.match_authors(data['pub_1'], data['ai_1'], data['pub_2'], data['ai_2'], params=data['params']))


@app.route('/init_scad_resources', methods=['POST'])
def init_scad_resources():
    lock=Lock()
    lock.acquire(blocking=True)
    sd = request.data.decode().strip()
    data = json.loads(sd)    

    # TODO Make this more flexible

    if "wombat_path" in data:
        print("Creating global Wombat connector from '%s'"%data['wombat_path'])
        classifier.CACHE['wombat'] = wb_conn(path = data['wombat_path'], create_if_missing = False, list_contents = True)

    if "english_stemmer" in data:
        print("Creating global Porter stemmer (English only) ['english_stemmer']")
        classifier.CACHE['english_stemmer'] = PorterStemmer(mode='NLTK_EXTENSIONS')

    if "english_stopwords" in data:
        print("Creating global stopword list for English ['english_stopwords']")
        classifier.CACHE['english_stopwords'] = set(stopwords.words('english')) 

    if "english_pretokenizer" in data:
        print("Creating global Moses tokenizer ['english_pretokenizer']")
        classifier.CACHE['english_pretokenizer'] = MosesTokenizer(lang='en')
    
    if "token_dblp_idf_path" in data:
        print("Creating global IDF resource from '%s' ['token_dblp_idf'] "%data['token_dblp_idf_path'])
        temp_idf={}
        with open(data['token_dblp_idf_path']) as infile:
            for line in infile:
                try:                (key, val) = line.strip().split("\t")
                except ValueError:  pass
                temp_idf[key] = float(val)
        classifier.CACHE['token_dblp_idf'] = temp_idf

    if "token_zbmath_idf_path" in data:
        print("Creating global IDF resource from '%s' ['token_zbmath_idf'] "%data['token_zbmath_idf_path'])
        temp_idf={}
        with open(data['token_zbmath_idf_path']) as infile:
            for line in infile:
                try:                (key, val) = line.strip().split("\t")
                except ValueError:  pass
                temp_idf[key] = float(val)
        classifier.CACHE['token_zbmath_idf'] = temp_idf
       
    if "stem_dblp_idf_path" in data:
        print("Creating global IDF resource from '%s' ['stem_dblp_idf'] "%data['stem_dblp_idf_path'])
        temp_idf={}
        with open(data['stem_dblp_idf_path']) as infile:
            for line in infile:
                try:                (key, val) = line.strip().split("\t")
                except ValueError:  pass
                temp_idf[key] = float(val)
        classifier.CACHE['stem_dblp_idf'] = temp_idf

    if "stem_zbmath_idf_path" in data:
        print("Creating global IDF resource from '%s' ['stem_zbmath_idf'] "%data['stem_zbmath_idf_path'])
        temp_idf={}
        with open(data['stem_zbmath_idf_path']) as infile:
            for line in infile:
                try:                (key, val) = line.strip().split("\t")
                except ValueError:  pass
                temp_idf[key] = float(val)
        classifier.CACHE['stem_zbmath_idf'] = temp_idf

    lock.release()
    return {}   # To make Flask happy ...


if __name__ == '__main__':
    app.run(host=sys.argv[1], port=sys.argv[2], debug=False)
