# Machine Learning : Going Deeper with Python and Theano

This repo contains the materials for my presentation to PyCon (SG), on 19 June 2015.

## Installation of requirements

Even if you like to ```virtualenv```, it may make sense to use system-wide
installations of some of the basic Python numerical packages, since 
they're likely to be ready-optimized (even if slightly older) than ones 
installed/compiled by ```pip``` (which will willingly install without 
```OpenBLAS```, for instance) : 

```
dnf install scipy numpy Cython
``` 

Then, as usual :

```
virtualenv env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 
# Wait...
```


### Git Repo friendly iPython Notebooks

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

