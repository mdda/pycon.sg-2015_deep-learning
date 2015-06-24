## Obtaining Data

These instructions can be used to create datafiles useful for 
characterwise learning (as required by ``ipynb/6-RNN-as-Author.ipynb``, for instance)


### Collection of Paul Graham Essays

Helpfully, there's a [https://github.com/rohitsm/PGEssays](GitHub repo) with the essays already downloaded.

To create a training file :

```
cd data # `pwd` == REPO/data
git clone https://github.com/rohitsm/PGEssays.git
mv PGEssays/content PG-content
rm -rf PGEssays

cat PG-content/* > PG-content.txt
```

This leaves a 2.4Mb file of Paul Graham generated text in ``./data/PG-content.txt``.


### Shakespeare's works

Helpfully, there's a [whole site](http://sydney.edu.au/engineering/it/~matty/Shakespeare/) 
dedicated to downloading Shakespeare's works.

To create a training file :

```
cd data # `pwd` == REPO/data

mkdir Shakespeare
cd Shakespeare/
wget http://sydney.edu.au/engineering/it/~matty/Shakespeare/shakespeare.tar.gz
tar -xzf shakespeare.tar.gz 
cd ..

cat Shakespeare/comedies/* Shakespeare/histories/* Shakespeare/tragedies/* > Shakespeare.plays.txt
cat Shakespeare/poetry/* > Shakespeare.poetry.txt
# NB: There are other folders here, but these are the simplest example sets

rm -rf Shakespeare/
```

This gives us two files : 

*  268Kb of Shakespeare's poetry in ``./data/Shakespeare.poetry.txt``
*  5.0Mb of Shakespeare's plays in  ``./data/Shakespeare.plays.txt``

