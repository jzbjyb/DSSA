# DSSA
Implementations of **D**ocument **S**equence with **S**ubtopic **A**ttention (DSSA) model described in the paper:

"Learning to Diversify Search Results via Subtopic Attention" <br/>
Zhengbao Jiang, Ji-Rong Wen, Zhicheng Dou, Wayne Xin Zhao, Jian-Yun Nie, and Ming Yue.

## Quick Start

The project is implemented using python 3.5 and tested in Linux environment.
Follow the steps to quickly run the model:

```shell
$ cd DSSA # enter project root directory
$ virtualenv -p /path/to/python3.5_interpreter dssa_env # build virtual environment using python3.5
$ source dssa_env/bin/activate # activate virtual environment
$ pip install -r etc/requirements.txt # install required packages
$ python run.py # train and test the model based on a small dataset
```

Because the model is trained only on 10 queries and tested on 3 queries,
you can see the final results in just a few minutes.
The console output is like:

```
EPO[1_0.2]	  train:0.279:544.187	   test:0.349
EPO[2_0.1]	  train:0.284:525.875	   test:0.351
EPO[3_0.1]	  train:0.293:510.472	   test:0.351
...
```

where `0.279` is the pair classification accuracy, `544.187` is the log loss,
`0.349` is the &#945;-nDCG of the test queries.

## How To Reproduce Experimental Results

You need first download the required data (use `python run.py -h` to see details of the required inputs):

```shell
$ cd DSSA # enter project root directory
$ wget http://www.playbigdata.com/dou/DSSA/data_cv.tar.gz # download data
$ tar xzvf data_cv.tar.gz # uncompress
```

Then run the model using downloaded data:

```shell
$ python run.py --cv --train_sample_path=data_cv/train_sample.data \
$                    --test_sample_path=data_cv/test_sample.data \
$                    --rel_feat_path=data_cv/rel_feat.csv \
$                    --doc_emb_path=data_cv/doc.emb \
$                    --query_emb_path=data_cv/query.emb
```

On our 24 core CPU machine, it takes roughly one day to complete the cross validation with final &#945;-nDCG around 0.45.

## How To Run On Your Dataset

Basically, your need to specify two things: **(1) several input files (2) the model configuration.**

The required input files can be seen by `python run.py -h`. The name of each command line arg is self-explanatory.
The model configuration (`Config` class in `run.py`) must by set manually.

### 1. input file format

1. `train_sample_path`: list-pairwise sample file of the following format:
```
{
  query_id_1: [([d1, d2], [(d3, d4, w34), (d5, d4, w45)]), ...],
  query_id_2: [([], [(d6, d7, w67)]), ...],
  ...
}
```
The first sample of `query_id_1` means that under context `[d1, d2]` we have two pairs: `d4 > d3` and `d4 > d5`.
You can use the following code to check file `data/train_sample.data` to better understand the format:
```python
import pickle
train_sample = pickle.load(open('data/train_sample.data', 'rb'))
for qid in train_sample:
  print('query is {}'.format(qid))
  for sample in train_sample[qid]:
    print('context is {}'.format(sample[0]))
    for pair in sample[1]:
      print('pair is {}>{} with weight {}'.format(pair[1], pair[0], pair[2]))
      input('press enter to continue')
```
2. `test_sample_path`: a dict with query id as key and candidate documents list as value.

You can use the following code to check file `data/test_sample.data` to better understand the format:
```python
import pickle
test_sample = pickle.load(open('data/test_sample.data', 'rb'))
for qid in test_sample:
  print('query is {}'.format(qid))
  print('candidate docs are {}'.format(test_sample[qid]))
```
3. `query_suggestion_path`: a xml file containing suggestions for each query.
  Check `data/query_suggestion.xml` for details.
4. `rel_feat_path`: a csv file containing relevance features for each query-doc pair.
  Check `data/rel_feat.csv` for details.
  The first two columns are *query* and *doc*, and the remaining columns are relevance features.
5. `doc_emb_path`: a file of which each line is a doc embedding with the format `doc_id v1 v2 ... vn`
  (`\t` as separation). Check `data/doc.emb` for details.
6. `query_emb_path`: a file containing embeddings for all queries (and their subtopics) which is similar to `doc_emb_path`.
  Check `data/query.emb` for details.
7. `save_model_path`: If set, model will be saved to this file.
8. `reuse_model_path`: If set, model will be loaded from this file.

### 2. generate list-pairwise train samples

To generate list-pairwise samples, you need TREC run files (baseline ranking),
offical Web Treck topic files, and diversity judgement files. We already provide you with these files in
`data/baseline_run`, `data/wt_topics`, and `data/wt_judge` folders respectively.
We use 4 Web Track ranging from 2009 to 2012 and the baseline rankings are generated from indri online service.
Run the following to generate train samples:
```shell
$ python prep.py 1 20 5 train_sample.data # use top 20 docs and 5 negative random permutations
```

### 3. evaluation

The `DiversityQuery` object in `type.py` is designed to calculate
diversity metrics (such as &#945;-nDCG and ERR-IA). In order to calcuate &#945;-nDCG,
you need both current ranking and global best ranking as normalization. These can be obtained by:
```shell
$ python prep.py 2 # generate DiversityQuery objects for baseline ranking
$ python prep.py 3 # generate DiversityQuery objects for global best ranking
```

### 4. model

The `DSSA` model is implemented using tensorflow. Moreover, it is scikit-learn compatible,
which means that you can use it as follow:
```python
dssa = DSSA(init_params)
X, y = ..., ...
dssa.fit(X, y)
```
If the model is saved, you can directly load it to do prediction or further training:
```python
dssa = DSSA(reuse_model='reuse_model_path', other_init_params)
testX = ...
ranks = dssa.predict(testX) # test the model
X, y = ..., ...
dssa.fit(X, y) # further training
```
Another benefit of being scikit-learn compatible is that
we can use `GridSearchCV` for automatizing parameter tuning:
```python
from sklearn.grid_search import GridSearchCV

dssa = DSSA(init_params)
X, y = [], []
for qid in all_queries:
  cur = {}
  cur['qid'] = qid
  cur['train_x'] = ... # list-pairwise training data for this query
  cur['test_x'] = ... # test data (candidate docs) for this query
  X.append(cur)
  y.append(dq) # dq is the DiversityQuery object for this query
tuned_params = {'learning_rate': [0.01, 0.1], 'n_epochs': [10, 20]}
gs = GridSearchCV(dssa, tuned_params, cv=2)
gs.fit(np.array(X), np.array(y))
print(gs.best_params_)
```
How to generate `train_x` and `test_x` and why we need these two distinct field is a little bit confusing.
I will try to explain this as clearly as possible.

The `X` and `y` must be indexed by query (i.e. `X.shape[0] = y.shape[0] = num_of_query`),
because in cross validation, we train and validate the model on distinct set of queries.
Training needs list-pairwise samples of a query, while validation (testing) only needs all the candidate docs of a query.
That's why we use two fields (`train_x` and `test_x`) in one query.

`test_x` is a `[n_candidate_docs, dim_of_each_doc]` numpy.ndarray, while
`train_x` is a `[n_sample, most_n_doc * dim_of_each_doc + most_n_pair * (2 * dim_of_each_doc + 2) + 2]` numpy.ndarray.

The dimension of `test_x` is easy to understand. `dim_of_each_doc = 1 + most_n_subtopic + 1`
which contains the index of the doc (the first `1`) and indexes of its subtopics (the `most_n_subtopic`).
Because different queries have different numbers of subtopics,
we need the last `1` to specify the number of the subtopics of this doc.
Actually, only the first column of `test_x` is different for each row because for the same query,
different docs share the same subtopics. The reason of this redundancy is for convenience of using RNN in tensorflow.

The dimension of `train_x` is hard to understand. In order to understand this, you need first make sure that
you know how list-pairwise sampling works. In list-pairwise sampling, 
a sample contains a context (previous selected docs) and a pair of docs.
The basic idea is that we organize all samples with the same context in one row (for the sake of efficiency).
So `n_sample` is the number of the unique contexts for a query;
`most_n_doc` is the maximun length of a context;
`most_n_pair` is the maximun number of pairs of a context.
The second last `2` is for pair preference judgement (0 or 1) and pair weight.
The last `2` specifies the number of docs in the context and the number of pairs.

# Reference

```
@inproceedings{Jiang:17SIGIR:DSSA,
  author = {Jiang, Zhengbao and Wen, Ji-Rong and Dou, Zhicheng and Zhao, Wayne Xin and Nie, Jian-Yun and Yue, Ming},
  title = {Learning to Diversify Search Results via Subtopic Attention},
  booktitle = {Proceedings of the 40th SIGIR},
  year = {2017},
}
```
The query suggestions and baseline runs can be downloaded from
[http://www.playbigdata.com/dou/hdiv/][link:hdiv]

The data required to reproduce the experimental results can be downloaded from
[http://www.playbigdata.com/dou/DSSA/][link:dssa]

For any issues with the code, feel free to contact `rucjzb AT 163.com`


[link:hdiv]: http://www.playbigdata.com/dou/hdiv/
[link:dssa]: http://www.playbigdata.com/dou/DSSA/
