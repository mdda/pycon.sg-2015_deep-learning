# Machine Learning : Going Deeper with Python and Theano

This repo contains the materials for my presentation to PyCon (SG), on 19 June **2015**.

If you are interested in the wonderful Virtual Machine goodness of my 2016 PyCon presentation,
the repo you should be looking at is [this one](https://github.com/mdda/deep-learning-workshop).

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
dnf install pydot
```

Note that installing ``python-ipython-notebook`` system-wide doesn't seem to work well,
because there is a version conflict involving ``tornado``. 

Then, as usual (but making use of these system-site-packages) :

```
virtualenv --system-site-packages env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 

# Wait 10mins (107Mb of stuff in env/lib/python2.7/site-packages/)
```

## Running the Presentation 

Starting at the live-plotting example (which will also want ``bokeh-server`` running, see below) :

```
. env/bin/activate
ipython notebook ipynb/1-LivePlotting.ipynb

# Then open a browser at : http://localhost:8888/
# or, more specifically  : http://localhost:8888/ipynb/1-LivePlotting.ipynb
```

If you have a browser already running, it may be best to use the ``--browser`` option to prevent the distracting launch of an additional browser window:
```
ipython notebook --port=8888 --browser=none
```

To run the live-plotting example, you'll also need to start the 
``bokeh-server`` in another process :

```
. env/bin/activate
bokeh-server
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

### GPU-aware iPython

#### Laptop 

Making use of an Nvidia card in a notebook (where ``COMMAND-LINE`` is 
the ``python`` command that one would ordinarily run) is usually as follows ::

```
THEANORC=theano.cuda-gpuarray.rc optirun {COMMAND-LINE}
```

However, because IPython spawns sub-processes to handle each kernel/notebook,
the ``optirun`` invocation isn't made for the child processes that should actually
perform the work on the GPU.  

So far, the only route to making this work has been to replace the 
``env/bin/python2`` with a script that runs ``optirun python2-bin`` where 
``python2-bin`` is a copy of the previously existing ``python2``.  But doing this
the causes all python (in that ``virtualenv``) to switch the GPU on, which 
wasn't really the plan.



### Notes : Git-friendly iPython Notebooks

Using the code from : http://pascalbugnion.net/blog/ipython-notebooks-and-git.html (and
https://gist.github.com/pbugnion/ea2797393033b54674af ), 
you can enable this kind of feature just on one repository, 
rather than installing it globally, as follows...

Within the repository, run : 
```
# Set the permissions for execution :
chmod 754 ./bin/ipynb_optional_output_filter.py

git config filter.dropoutput_ipynb.smudge cat
git config filter.dropoutput_ipynb.clean ./bin/ipynb_optional_output_filter.py
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

There are two different approaches to doing the cleansing in the ``REPO/bin`` directory : 

* ``ipynb_output_filter.py`` : which is probably more comprehensive, since it uses iPython itself
  to parse and output the notebooks - but care must be taken to ensure that it is run within the current ``env``
  
* ``ipynb_optional_output_filter.py`` : This is my current chosen approach, which only uses
  ``import json`` to parse the notebook files (and so can be executed as a plain script).  It 
  also includes the ``git:suppress_outputs=false`` option that might be useful...

To include disable the output-cleansing feature in a notebook in the latter case, 
simply add to its metadata (Edit-Metadata) as a first-level entry (``true`` is the default): 

```
  "git" : { "suppress_outputs" : false },
```

### Notes : Building the Presentation

For 'blocks-introduction-mnist.ipynb' I used the tutorial 
from the ``blocks`` documentation as a starter :

* ```wget https://raw.githubusercontent.com/mila-udem/blocks/master/docs/tutorial.rst```
* ```pandoc --mathjax --from=rst --to=markdown_mmd tutorial.rst > tutorial.md```

Also useful :

* [MathJax](http://nbviewer.ipython.org/github/olgabot/ipython/blob/master/examples/Notebook/Typesetting%20Math%20Using%20MathJax.ipynb)
* [Bokeh](http://bokeh.pydata.org/en/latest/docs/quickstart.html)



### Notes : Installing PyGPU

On the date of the PyCon, building the ``libgpuarray`` library from github FAILS : 
```
NOW DONE - submitted a PR for : 
           https://github.com/Theano/libgpuarray/issues/55
           """ gcc 5.1.1 : max_align_t also defined in stddef.h """
```

There is a full write-up on how to install an Nvidia GPU under Fedora 22 
as a [blog posting](http://blog.mdda.net/oss/2015/07/07/nvidia-on-fedora-22/)

And an additional write-up for the case that you're installing to a laptop
with 'dual graphics cards' 
in [this blog post](http://blog.mdda.net/oss/2015/06/20/nvidia-on-fedora-22-laptop/).
