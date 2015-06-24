## Obtaining Data

### Collection of Paul Graham Essays

Helpfully, there's a [https://github.com/rohitsm/PGEssays](GitHub repo) with the essays already downloaded.

To create a useful characterwise training (as required by ``ipynb/6-RNN-as-Author.ipynb``, for instance) :

```
cd data # `pwd` == REPO/data
git clone https://github.com/rohitsm/PGEssays.git
mv PGEssays/content PG-content
rm -rf PGEssays

cat PG-content/* > PG-content.txt
```

This leaves a 2.4Mb file of Paul Graham generated text in ``./data/PG-content.txt``.





