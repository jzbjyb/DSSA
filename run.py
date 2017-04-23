#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle, random, operator
from functools import reduce
from dssa import DSSA
from prep import load_query_suggestion, load_emb
from type import DiversityQuery
SEED = 2017
random.seed(SEED)
np.random.seed(SEED)

flags = tf.flags
flags.DEFINE_string('train_sample_path', 'data/train_sample.data', 'list-pairwise sample file')
flags.DEFINE_string('test_sample_path', 'data/test_sample.data', 'test sample file')
flags.DEFINE_string('query_suggestion_path', 'data/query_suggestion.xml', 'query suggestion file')
flags.DEFINE_string('rel_feat_path', 'data/rel_feat.csv', 'relevance feature file')
flags.DEFINE_string('doc_emb_path', 'data/doc.emb', 'document embedding file')
flags.DEFINE_string('query_emb_path', 'data/query.emb', 'query & subquery embedding file')
flags.DEFINE_string('save_model_path', 'model', 'model storage path')
flags.DEFINE_string('reuse_model_path', None, 'loading model path')

FLAGS = flags.FLAGS

class Config(object):
    cell_type = 'vanilla' # rnn cell type ('vanilla', 'LSTM', 'GRU')
    interaction = 'general' # interaction type ('general', 'dot')
    n_rel_feat = 18 # number of relevance feature
    n_doc_emb = 25 # dimension of document embedding
    n_query_emb = 25 # dimension of query embedding
    hidden_size = 10 # rnn hidden layer size
    most_n_doc = None # maximum context document sequence length (don't need to set manually, will be calculated from train samples)
    most_n_pair = None # maximum pair number per context (don't need to set manually, will be calculated from train samples)
    most_n_subtopic = None # maximum subtopic number per query (don't need to set manually, will be calculated from train samples)
    lambdaa = 0.5 # trade-off between relevance score and diversity score
    learning_rate = 0.001 # learning rate
    n_epochs = 50 # iteration number

def main(_):
    # load data
    config = Config()
    suggestion = load_query_suggestion(FLAGS.query_suggestion_path)
    rel_feat = pd.read_csv(FLAGS.rel_feat_path)
    doc_emb = load_emb(FLAGS.doc_emb_path)
    query_emb = load_emb(FLAGS.query_emb_path)
    samples = pickle.load(open(FLAGS.train_sample_path, 'rb'))
    test_samples = pickle.load(open(FLAGS.test_sample_path, 'rb'))
    DiversityQuery.load_alpha_nDCG_global_best('data/best_alpha_nDCG.data')  # used as the normalization of alpha-nDCG
    ts = pickle.load(open('data/test_sample_dq.data', 'rb')) # the doc sequence should be the same as test_samples
    all_samples = []
    for qid in samples:
        for s in samples[qid]:
            all_samples.append(s)
    config.most_n_doc, config.most_n_pair = \
        reduce(lambda t, x: [max(t[0], len(x[0])), max(t[1], len(x[1]))], all_samples, [0, 0])
    config.most_n_subtopic = max([len(suggestion[qid][1]) for qid in suggestion]) + 1
    print('most_n_doc: {}, most_n_pair: {}, most_n_subtopic: {}'.format(config.most_n_doc, config.most_n_pair, config.most_n_subtopic))
    # pre-processing
    rel_feat_names = list(sorted(set(rel_feat.columns) - {'query', 'doc'}))
    rel_feat[rel_feat_names] = StandardScaler().fit_transform(rel_feat[rel_feat_names])
    rel_feat = dict(zip(map(lambda x: tuple(x), rel_feat[['query', 'doc']].values), rel_feat[rel_feat_names].values.tolist()))
    # data assemble
    train_data, test_data = [], []
    DOC_DICT = {}
    SUBTOPIC_DICT = {}
    SUBTOPIC = {}
    SUBTOPIC_EMB_DICT = {}
    def get_doc_emb(doc_emb, rel_feat, config):
        emb = []
        for d in sorted(DOC_DICT.items(), key=operator.itemgetter(1)):
            qid, docid = d[0].split('_')
            rel = []
            for subq in SUBTOPIC[qid]:
                rel.append(rel_feat[(subq, docid)])
            rel.append([0] * config.n_rel_feat * (config.most_n_subtopic - len(SUBTOPIC[qid])))
            emb.append(np.concatenate(rel + [doc_emb[docid]]))
        return np.vstack(emb)
    def get_subtopics(qid, suggestion, query_emb, config):
        query = suggestion[qid][0]
        subtopics = [sug[0] for sug in suggestion[qid][1]]
        # put the query in the first place
        subtopics = [query] + sorted(set(subtopics) - {query})[:config.most_n_subtopic - 1]
        SUBTOPIC[qid] = subtopics
        for st in subtopics:
            SUBTOPIC_EMB_DICT[st] = query_emb[st] + [1 / max(len(subtopics) - 1, 1)] # init weight
        return subtopics
    def data_assemble(qid, subtopics, context, pairs, config):
        input_dim = 1 + config.most_n_subtopic + 1
        data = np.zeros((config.most_n_doc * input_dim + config.most_n_pair * (2 * input_dim + 2) + 2))
        if len(context) > 0:
            doc_part = np.vstack([[DOC_DICT.setdefault('{}_{}'.format(qid, context[i]), len(DOC_DICT))] for i in range(len(context))])
            subtopic_part = np.tile([SUBTOPIC_DICT.setdefault(q, len(SUBTOPIC_DICT)) for q in subtopics], [len(context)]).reshape(len(context), -1)
            datap = np.zeros((len(context), input_dim))
            datap[:, :1 + len(subtopics)] = np.concatenate([doc_part, subtopic_part], axis=1)
            datap[:, -1] = len(subtopics)
            data[:len(context) * input_dim] = datap.reshape(-1)
        datap = np.zeros((len(pairs), 2 * input_dim + 2))
        rand = (np.random.rand(len(pairs)) > 0.5).astype(np.int)
        p1_part = np.vstack([[DOC_DICT.setdefault('{}_{}'.format(qid, pairs[i][rand[i]]), len(DOC_DICT))]
                             for i in range(len(pairs))])
        p2_part = np.vstack([[DOC_DICT.setdefault('{}_{}'.format(qid, pairs[i][1 - rand[i]]), len(DOC_DICT))]
                             for i in range(len(pairs))])
        weight_pair = np.array([pairs[i][2] for i in range(len(pairs))])
        subtopic_part = np.tile([SUBTOPIC_DICT.setdefault(q, len(SUBTOPIC_DICT)) for q in subtopics], [len(pairs)]).reshape(len(pairs), -1)
        datap[:, :1] = p1_part
        datap[:, 1:1 + len(subtopics)] = subtopic_part
        datap[:, input_dim - 1] = len(subtopics)
        datap[:, input_dim:input_dim + 1] = p2_part
        datap[:, input_dim + 1:input_dim + 1 + len(subtopics)] = subtopic_part
        datap[:, 2 * input_dim - 1] = len(subtopics)
        datap[:, -1] = weight_pair
        datap[:, -2] = 1 - rand
        data[config.most_n_doc * input_dim:config.most_n_doc * input_dim + np.product(datap.shape)] = datap.reshape(-1)
        data[-2] = len(context)
        data[-1] = len(pairs)
        return data
    def data_assemble_test(qid, subtopics, docs, config):
        data = np.zeros((len(docs), 1 + config.most_n_subtopic + 1))
        doc_part = np.vstack([[DOC_DICT.setdefault('{}_{}'.format(qid, docs[i]), len(DOC_DICT))] for i in range(len(docs))])
        subtopic_part = np.tile([SUBTOPIC_DICT.setdefault(q, len(SUBTOPIC_DICT)) for q in subtopics], [len(docs)]).reshape(len(docs), -1)
        data[:, :1 + len(subtopics)] = np.concatenate([doc_part, subtopic_part], axis=1)
        data[:, -1] = len(subtopics)
        return data
    print('start data assemble ...')
    for qid in sorted(samples.keys()):
        subtopics = get_subtopics(qid, suggestion, query_emb, config)
        data_per_query = dict([('qid', qid), ('query', subtopics[0]), ('subtopics', subtopics)])
        datap = [data_assemble(qid, subtopics, s[0], s[1], config) for s in samples[qid]]
        data_per_query['train_x'] = np.vstack(datap) if len(datap) > 0 else\
            np.zeros((0, config.most_n_doc * (1 + config.most_n_subtopic + 1) + config.most_n_pair * (2 * (1 + config.most_n_subtopic + 1) + 2) + 2))
        train_data.append(data_per_query)
    for qid in sorted(test_samples.keys()):
        subtopics = get_subtopics(qid, suggestion, query_emb, config)
        data_per_query = dict([('qid', qid), ('query', subtopics[0]), ('subtopics', subtopics)])
        data_per_query['test_x'] = data_assemble_test(qid, subtopics, test_samples[qid], config)
        test_data.append(data_per_query)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    print('end data assemble ...')
    doc_emb_raw = get_doc_emb(doc_emb, rel_feat, config)
    query_emb_raw = np.array([SUBTOPIC_EMB_DICT[i[0]] for i in sorted(SUBTOPIC_DICT.items(), key=operator.itemgetter(1))])
    dssa = DSSA(n_rel_feat=config.n_rel_feat, n_doc_emb=config.n_doc_emb, n_query_emb=config.n_query_emb, hidden_size=config.hidden_size, interaction=config.interaction,
                cell_type=config.cell_type, lambdaa=config.lambdaa, most_n_subquery=config.most_n_subtopic, most_n_pair=config.most_n_pair, most_n_doc=config.most_n_doc,
                doc_emb=doc_emb_raw, query_emb=query_emb_raw, learning_rate=config.learning_rate, n_epochs=config.n_epochs, batch_size=50, optimization='listpair',
                save_epochs=1, verbose=True, random_seed=SEED, reuse_model=FLAGS.reuse_model_path, save_model=FLAGS.save_model_path)
    for e in dssa.fit_iterable(train_data):
        # the order of the doc sequence in both test_data and ts (list of DiversityQuery object) should be the same
        ranks = dssa.predict(test_data)
        metrics = []
        for i in range(len(test_data)):
            dq = ts[test_data[i]['qid']]
            ndq = DiversityQuery(dq.query, dq.qid, dq.subtopics, dq.docs[ranks[i]])
            metrics.append(ndq.get_metric('alpha_nDCG'))
        print('\t{:>7}:{:>5.3f}'.format('test', np.mean(metrics)), end='', flush=True)

if __name__ == '__main__':
    tf.app.run()