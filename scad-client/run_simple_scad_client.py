import argparse
from scad_client import simple_scad_client

def run_simple_scad_client(pargs):

    client = simple_scad_client(scad_server_url=pargs.scad_url, logfile=pargs.logfile, verbose=pargs.verbose)
    client.load_publications(pub_file_name=pargs.pubfile, blocking_pattern=pargs.blocking_pattern)
    client.load_resources(pargs.resourcefile)
    client.load_params(pargs.paramfile)
    
    client.match_publications(evaluate=pargs.evaluate, name_matching_method=pargs.name_matching_method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scad_url', help='scad server url', required=True)
    parser.add_argument('--pubfile', help='List of publications in json format', required=True)
    parser.add_argument('--blocking_pattern', help='Author names must match this RE pattern to be considered at all', required=False, default='')

    parser.add_argument('--logfile', help='Log output file', required=False, default='')
    parser.add_argument('--paramfile', help='Matching features as JSON', required=True)
    parser.add_argument('--resourcefile', help='Resources as JSON', required=True)    
    parser.add_argument('--name_matching_method', help='Method to match potentially ambiguous author names', required=True)
    parser.add_argument('--evaluate', dest='evaluate', help='', action='store_true')
    parser.add_argument('--verbose', dest='verbose', help='', action='store_true')
    run_simple_scad_client(parser.parse_args())









