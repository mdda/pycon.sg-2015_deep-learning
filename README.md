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
# or, more specifically  : http://localhost:8888/notebooks/Deep-Learning-with-Blocks-and-Theano.ipynb

## Another suggestion (--browser prevents the distracting launch of an browser window)
ipython notebook --port=8888 --browser=none
```

To run the live-plotting example, you'll also need to start the 
``bokeh-server`` in another process :

```
. env/bin/activate
bohek-server
```

### Running the Presentation (across a network)

```
. env/bin/activate
ipython notebook --ip=0.0.0.0 --port=8888 --browser=none &
bokeh-server --ip=0.0.0.0 --port=8889
```
The iPython notebook call to ``bokeh.io.output_notebook()`` appears 
to find the correct port for the ``bokeh-server`` automagically.

Remember to adjust the firewall to allow these two open ports...



### Notes : Git-friendly iPython Notebooks

Using the code from : http://pascalbugnion.net/blog/ipython-notebooks-and-git.html (and
https://gist.github.com/pbugnion/ea2797393033b54674af ), 
you can enable the feature just on one repository, 
rather than installing it globally, as follows...

Within the repository, run : 
```
# Set the permissions for execution :
chmod 754 ./bin/ipynb_optional_output_filter.py

git config filter.dropoutput_ipynb.smudge cat
git config filter.dropoutput_ipynb.clean ./bin/ipynb_optional_output_filter.py"
```
this will add suitable entries to ``./.git/config``.

or, alternatively, create the entries manually by ensuring that your ``.git/config`` includes the lines :
```
[filter "dropoutput_ipynb"]
	smudge = cat
	clean = ./bin/ipynb_output_filter.py
```
(where ``REPO`` is the absolute path to the root of the checked out repository).


Note also that there's a ``<REPO>/.gitattributes`` file here containing the following:
```
*.ipynb    filter=dropoutput_ipynb
```

### Notes : Building the Presentation

For 'blocks-introduction-mnist.ipynb' I used the tutorial 
from the ``blocks`` documentation as a starter :

* ```wget https://raw.githubusercontent.com/mila-udem/blocks/master/docs/tutorial.rst```
* ```pandoc --mathjax --from=rst --to=markdown_mmd tutorial.rst > tutorial.md```

Also useful :

* [MathJax](http://nbviewer.ipython.org/github/olgabot/ipython/blob/master/examples/Notebook/Typesetting%20Math%20Using%20MathJax.ipynb)
* [Bokeh](http://bokeh.pydata.org/en/latest/docs/quickstart.html)



