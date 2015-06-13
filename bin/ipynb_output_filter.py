# Runs using command-line python (which should inherit ./env)

import sys

version = None

debug=None
debug=open('log-ipynb-filter','w+')

if debug:
    debug.write("Filterering\n")

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')

try:
    # Jupyter
    from jupyter_nbformat import reads, write
except ImportError:
    try:
        # New IPython
        from IPython.nbformat import reads, write
    except ImportError:
        # Old IPython
        from IPython.nbformat.current import reads, write
        version = 'json'

to_parse = sys.stdin.read()

if not version:
    import json
    json_in = json.loads(to_parse)
    version = json_in['nbformat']

if debug:
    debug.write("nbformat=%s\n" % (version))
    
json_in = reads(to_parse, version)

if hasattr(json_in, 'worksheets'):
    # IPython
    sheets = json_in.worksheets
else:
    # Jupyter
    sheets = [json_in]

for sheet in sheets:
    for cell in sheet.cells:
        for field in ("outputs", ):
            if field in cell:
                if debug:
                    debug.write("deleting 'sheets.sheet.cells.%s' : %s\n" % (field, cell[field]) )
                cell[field] = []
        for field in ("prompt_number", "execution_number", ):
            if field in cell:
                del cell[field]
        for field in ("execution_count", ):
            if field in cell:
                cell[field] = None

if 'signature' in json_in.metadata:
    json_in.metadata['signature'] = ""

write(json_in, sys.stdout, version)

if debug:
    write(json_in, debug, version)
    debug.write("json-result :\n%s" % (json_in))
    debug.write("Finished\n")

if debug:
  debug.close()

exit(0)
