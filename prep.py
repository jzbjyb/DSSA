#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, pickle
import numpy as np
from xml.etree.ElementTree import parse
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from collections import defaultdict
from type import DiversityQuery, DiversityDoc

def load_emb(filename, sep='\t'):
    '''
    load embedding from file
    :param filename: embedding file name
    :param sep: the char used as separation symbol
    :return: a dict with item name as key and embedding vector as value
    '''
    with open(filename, 'r') as fp:
        result = {}
        for l in fp:
            l = l.strip()
            if l == '':
                continue
            sp = l.split(sep)
            vals = [float(sp[i]) for i in range(1, len(sp))]
            result[sp[0]] = vals
        return result

def load_query_suggestions(filenames):
    '''
    load query suggestions from multiple files
    :param filenames: a list of file names
    :return: a dict with qid as key and a tree-like structure as value
    '''
    r = {}
    for f in filenames:
        r.update(load_query_suggestion(f))
    return r

def load_query_suggestion(filename):
    '''
    load query suggestions from file
    :param filename: query suggestion file name
    :return: a dict with qid as key and a tree-like structure as value
    '''
    result = {}
    doc = parse(filename)
    nums = [0, 0]
    for query in doc.iterfind('//google/suggestions/topic'):
        qid = query.get('number')
        query_str = query.findtext('query')
        result[qid] = (query_str, [])
        has = False
        for sug1 in query.iterfind('subtopic1'):
            has = True
            nums[0] += 1
            result[qid][1].append((sug1.findtext('suggestion'), []))
            for sug2 in sug1.iterfind('subtopic2'):
                nums[1] += 1
                result[qid][1][-1][1].append((sug2.text, []))
        if not has: # at least has one subquery which is the query itself
            result[qid][1].append((query_str, []))
    print('{} 1-level suggestion, {} 2-level suggestion'.format(*nums))
    return result

def load_rank_docs(filename, top=100):
    '''
    load rank from run file
    :param filename: run file name, the format of the file is consistent with TREC,
        see http://trec.nist.gov/ for more info
    :param top: number of top docs
    :return: a dict with query as key and doc sequence as value
    '''
    result = defaultdict(lambda: [])
    with open(filename, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l == '':
                continue
            q, _, docid, _, _, _ = l.split(' ')
            if len(result[q]) >= top:
                continue
            result[q].append(docid)
    return result

def load_datasets(filenames, qrels=None, topics_xmls=None, top=100):
    '''
    load run, judgement, and official queries
    :param filenames: run files
    :param qrels: diversity judgement files
    :param topics_xmls: official query (topic) files
    :param top: number of top docs
    :return: queries (a list of DiversityQuery object), topics, judgements
    '''
    queries, topics, judge = [None] * 3
    for i in range(len(filenames)):
        r = load_dataset(filenames[i], qrels[i] if qrels else None,
                         topics_xmls[i] if topics_xmls else None, top=top)
        queries = r[0] if not queries else queries + r[0]
        if not topics:
            topics = r[1]
        else:
            topics.update(r[1])
        if not judge:
            judge = r[2]
        else:
            judge.update(r[2])
    return queries, topics, judge

def load_dataset(filename, qrel=None, topics_xml=None, top=100):
    '''
    load run, judgement, and official queries
    :param filenames: run file
    :param qrels: diversity judgement file
    :param topics_xmls: official query (topic) file
    :param top: number of top docs
    :return: queries (a list of DiversityQuery object), topics, judgements

    '''
    result = load_rank_docs(filename, top=500)
    if qrel is not None and topics_xml is not None:
        topics, judge = load_TREC_diversity(qrel, topics_xml)
        result_with_judge = []
        for q, docs in result.items():
            remain_docs = [(d, judge[q][d].keys()) for d in docs][:top]
            result_with_judge.append((q, remain_docs))
        queries = judgedict_to_query(result_with_judge, topics)
        return queries, topics, judge
    return result

def load_TREC_diversity(qrel, topics_xml, restrict=None):
    '''
    load Web Track diversity judgement file and official query (topic) file
    :param qrel: diversity judgement file
    :param topics_xml: official query (topic) file
    :param restrict: docid restriction (used to only remain Category B documents)
    :return:
        queries (topics): a dict with topic id as key and 'query', 'description', 'type', 'subtopics' as value
            'subtopics' is a dict with subtopic id as key and 'description', 'type' as value
        judge: a dict with topic id as key and docid as keys in value,
            each docid corresponds to a dict of subtopic id, with relevance judgement as value
    '''
    # load topics and subtopics
    topics = defaultdict(lambda: {})
    tree = ET.ElementTree(file=topics_xml)
    for topic in tree.iterfind('topic'):
        qid = topic.attrib['number']
        topics[qid]['query'] = topic[0].text
        topics[qid]['description'] = topic[1].text
        topics[qid]['type'] = topic.attrib['type']
        topics[qid]['subtopics'] = {}
        for subtopic in topic.iter(tag='subtopic'):
            sid = subtopic.attrib['number']
            topics[qid]['subtopics'][sid] = {'type': subtopic.attrib['type'], 'description': subtopic.text}
        if len(topics[qid]['subtopics']) == 0: # if no subquery, just treat query as the only one subquery
            topics[qid]['subtopics']['0'] = {'type': 'raw', 'description': topics[qid]['query']}
    print('load topics from [{}]'.format(topics_xml))
    # load judgement
    judge = defaultdict(lambda: defaultdict(lambda: {}))
    with open(qrel, 'r') as fp:
        cqid, cdocid = [None]*2
        ctopic = {}
        for l in fp:
            l = l.strip()
            if l == '':
                continue
            qid, topic, docid, rel = l.split()
            rel = int(rel)
            if restrict and docid[10:16] not in restrict:
                continue
            if cqid == qid and cdocid == docid:
                if rel != 0 and rel != -2:
                    ctopic[topic] = rel
            else:
                # save one doc
                if cqid is not None:
                    judge[cqid][cdocid] = ctopic
                cqid, cdocid = qid, docid
                ctopic = {}
                if rel != 0 and rel != -2:
                    ctopic[topic] = rel
        # save last doc
        if cqid is not None:
            judge[cqid][cdocid] = ctopic
    print('load judgement from [{}]'.format(qrel))
    unique_docs = set(np.concatenate([list(qv.keys()) for qk,qv in judge.items()]))
    unique_query_docs = set(np.concatenate([[qk+'#'+k for k in list(qv.keys())]
                                            for qk,qv in judge.items()]))
    unique_labeled_docs = set(np.concatenate([[dk for dk,dv in qv.items() if len(dv)>0]
                                              for qk,qv in judge.items()]))
    print('totally {} queries, {} unique docs, {} unique query-docs, {} unique labeled docs'.
          format(len(topics), len(unique_docs), len(unique_query_docs), len(unique_labeled_docs)))
    return topics, judge

def judgedict_to_query(ranks, topics):
    '''
    generate a list of DiversityQuery object
    :param ranks: a list of ranks
    :param topics: query (topic) generated from official fficial query (topic) file
    :return: a list of DiversityQuery object
    '''
    result = []
    for qk,qv in ranks:
        dq = DiversityQuery(topics[qk]['query'], qk, set(topics[qk]['subtopics'].keys()),
                            [DiversityDoc(dk, set(dv)) for dk,dv in qv])
        result.append(dq)
    return result

if __name__ == '__main__':
    OP = int(sys.argv[1])
    judge_queries, topics, judge = load_datasets(['data/baseline_run/wt2009.txt',
                                                  'data/baseline_run/wt2010.txt',
                                                  'data/baseline_run/wt2011.txt',
                                                  'data/baseline_run/wt2012.txt'],
                                                 ['data/wt_judge/2009.diversity.qrels',
                                                  'data/wt_judge/2010.diversity.qrels',
                                                  'data/wt_judge/2011.diversity.qrels',
                                                  'data/wt_judge/2012.diversity.qrels'],
                                                 ['data/wt_topics/wt2009.topics.xml',
                                                  'data/wt_topics/wt2010.topics.xml',
                                                  'data/wt_topics/wt2011.topics.xml',
                                                  'data/wt_topics/wt2012.topics.xml'])
    if OP == 1:
        '''
        generate list-pairwise training samples
        three command line arg are desired, which are:
        top doc num, random permutation num, output file name
        '''
        MAXDOC, PERM_NUM, OUT, = sys.argv[2:]
        MAXDOC = int(MAXDOC)
        PERM_NUM = int(PERM_NUM)
        sam = {}
        for dq in judge_queries:
            print('generate samples for query {}'.format(dq.qid))
            if len(dq.docs) < MAXDOC:
                continue
            sam[dq.qid] = []
            basedq = dq.top(MAXDOC).best_rank('alpha_nDCG', top_n=MAXDOC)
            sam[dq.qid] = DiversityQuery\
                .get_pair_samples(basedq, metric='alpha_nDCG', use_best_sample=True, perm_num=PERM_NUM)
        pickle.dump(sam, open(OUT, 'wb'), True)
    elif OP == 2:
        '''
        generate list of DiversityQuery objects which are convenient for calculating metric score
        '''
        result = {}
        for dq in judge_queries:
            if dq.qid in {'95', '100'}: # queries without diversity judgement
                continue
            result[dq.qid] = dq.top(50) # evaluate based on top 50 results
        pickle.dump(result, open('data/test_sample_dq.data', 'wb'), True)
    elif OP == 3:
        '''
        generate DiversityQuery objects of best rankings,
        which means that all docs with positive judgement are included
        the generated objects are used as the normalization when calculating alpha-nDCG
        '''
        best_result_with_judge = [(q, [(d, judge[q][d].keys()) for d in docs.keys() if len(judge[q][d].keys()) > 0])
                                  for q, docs in judge.items()]
        queries = judgedict_to_query(best_result_with_judge, topics)
        best_queries = []
        for dq in queries:
            if len(dq.docids) <= 0:
                continue
            best_queries.append(dq.best_rank(metric='alpha_nDCG'))
        pickle.dump(dict((q.qid, q) for q in best_queries), open('data/best_alpha_nDCG.data', 'wb'), True)