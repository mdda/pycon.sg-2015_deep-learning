# Machine Learning : Going Deeper with Python and Theano

This repo contains the materials for my presentation to PyCon (SG), on 19 June 2015.

## Installation of requirements

Even if you like to ```virtualenv```, it may make sense to use system-wide
installations of some of the basic Python numerical packages, since 
they're likely to be ready-optimized (even if slightly older) than ones 
installed/compiled by ```pip``` (which will willingly install without 
```OpenBLAS```, for instance) : 

```
dnf install scipy numpy python-pandas Cython 
```

(and, since it's so handy):

```
dnf install python-ipython-notebook
```


Then, as usual (but making use of these system-site-packages) :

```
virtualenv --system-site-packages env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 

# Wait 10mins (107Mb of stuff in env/lib/python2.7/site-packages/)
```

## Running the Presentation

```
. env/bin/activate
ipython notebook ipynb/Deep-Learning-with-Blocks-and-Theano.ipynb 
# Then open a browser at : http://localhost:8888/
# or, more specifically : http://localhost:8888/notebooks/Deep-Learning-with-Blocks-and-Theano.ipynb

## Another suggestion (--browser prevents the distracting launch of an browser window)
#ipython notebook --matplotlib inline --port=8888 --browser=false

```


### Notes : Git-friendly iPython Notebooks

Using the code from https://github.com/toobaz/ipynb_output_filter (which 
was referenced from http://stackoverflow.com/questions/18734739/using-ipython-notebooks-under-version-control), 
you can enable the feature just on one repository, 
rather than installing it globally, as follows...

Within the repository, run : 
```
git config filter.dropoutput_ipynb.smudge cat
git config filter.dropoutput_ipynb.clean 'python ./bin/ipynb_output_filter.py'
```

This will add suitable entries to ``./.git/config``.

Note also that there's a ``<REPO>/.gitattributes`` file here containing the following:
```
*.ipynb    filter=dropoutput_ipynb
```

### Notes : Building the Presentation

For 'Deep-Learning-with-Blocks-and-Theano.ipynb' I used the tutorial 
from the ``blocks`` documentation as a starter :

* ```wget https://raw.githubusercontent.com/mila-udem/blocks/master/docs/tutorial.rst```
* ```pandoc --mathjax --from=rst --to=markdown_mmd tutorial.rst > tutorial.md```

Also useful :

* [MathJax](http://nbviewer.ipython.org/github/olgabot/ipython/blob/master/examples/Notebook/Typesetting%20Math%20Using%20MathJax.ipynb)
* [Bokeh](http://bokeh.pydata.org/en/latest/docs/quickstart.html)



