# Machine Learning : Going Deeper with Python and Theano

This repo contains the materials for my presentation to PyCon (SG), on 19 June 2015.


### Git Repo friendly iPython Notebooks

Using the code from https://github.com/toobaz/ipynb_output_filter, you can
enable the feature just on one repository, rather than installing it globally, as follows...

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

