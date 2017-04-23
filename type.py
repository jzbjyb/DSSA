# -*- coding: utf-8 -*-

import numpy as np
import scipy
import pandas as pd
import pickle
from metric import ERR_IA, best_ERR_IA, alpha_nDCG, best_alpha_nDCG

class DiversityDoc:
    def __init__(self, docid, subtopics):
        self.docid = docid
        self.subtopics = set(subtopics)

class DiversityQuery:
    K = 20 # evaluation based on top 20 docs
    ALPHA = 0.5 # alpha of alpha_nDCG
    ALPHA_NDCG_GLOBAL_BEST = None

    @staticmethod
    def load_alpha_nDCG_global_best(filename):
        '''
        load alpha_nDCG normalization
        '''
        DiversityQuery.ALPHA_NDCG_GLOBAL_BEST = pickle.load(open(filename, 'rb'))

    @staticmethod
    def to_ranking_file(filename, dqlist, tag):
        '''
        output in TREC run format
        :param filename: output file name
        :param dqlist: list of DiversityQuery object
        :param tag: tag
        '''
        dqlist = sorted(dqlist, key=lambda dq: int(dq.qid))
        with open(filename, 'w') as fp:
            for dq in dqlist:
                for r in range(len(dq.docids)):
                    fp.write('{} {} {} {} {} {}\n'.format(dq.qid, 'Q0', dq.docids[r], r + 1, 1 / (r + 1), tag))

    def __init__(self, query, qid, subtopics, docs):
        self.query = query
        self.qid = qid
        self.subtopics = set(subtopics)
        self.docs = np.array(docs)
        self.docids = np.array([d.docid for d in docs])
        self._subtopics_map = {}
        for s in self.subtopics:
            self._subtopics_map.setdefault(s, len(self._subtopics_map))
        def array_set(length, indice, values):
            arr = np.zeros(length)
            arr[indice] = values
            return arr
        self._grade_matrix = np.array([array_set(len(self._subtopics_map), [self._subtopics_map[t] for t in d.subtopics], 1) for d in self.docs])

    def get_metric(self, metric='alpha_nDCG', normalize=True):
        '''
        get metric score
        :param metric: alpha_nDCG or ERR_IA
        :param normalize: whether to normalize
        :return: metric score
        '''
        if metric == 'ERR_IA':
            return self.ERR_IA()
        elif metric == 'alpha_nDCG':
            return self.alpha_nDCG(normalize=normalize)
        else:
            raise ValueError('unsupported metric {}'.format(metric))

    def to_dataframe(self, as_index=False):
        '''
        to pandas DataFrame
        :param as_index: whether use query and doc as index
        :return: a DataFrame object
        '''
        df = pd.DataFrame({'query': np.repeat(self.query, len(self.docs)), 'doc': list(map(lambda x: x.docid, self.docs))})
        if as_index:
            return df.set_index(['query', 'doc'])
        return df

    def ERR_IA(self):
        return ERR_IA(self._grade_matrix.T, k=self.K)

    def alpha_nDCG(self, normalize=False):
        k = min(self._grade_matrix.shape[0], self.K)
        if normalize:
            if self.ALPHA_NDCG_GLOBAL_BEST and self.qid in self.ALPHA_NDCG_GLOBAL_BEST:
                best_k = min(self.ALPHA_NDCG_GLOBAL_BEST[self.qid]._grade_matrix.shape[0], self.K)
                best = best_alpha_nDCG(self.ALPHA_NDCG_GLOBAL_BEST[self.qid]._grade_matrix.T, alpha=self.ALPHA, k=min(best_k, k))
            else:
                best = best_alpha_nDCG(self._grade_matrix.T, alpha=self.ALPHA, k=k)
            alpha_nDCG_normalize = alpha_nDCG(self._grade_matrix.T, alpha=self.ALPHA, k=k, normalization=best[0])
            if alpha_nDCG_normalize > 1:
                raise Exception('query {} nDCG normalization overflow'.format(self.qid))
            return alpha_nDCG_normalize
        else:
            return alpha_nDCG(self._grade_matrix.T, alpha=self.ALPHA, k=k)

    def top(self, top_n=None):
        '''
        return top_n docs
        :param top_n: number of top docs
        :return: a new DiversityQuery object with top_n docs
        '''
        top_n = top_n or len(self.docs)
        return DiversityQuery(self.query, self.qid, self.subtopics, self.docs[:top_n])

    def best_rank(self, metric='alpha_nDCG', top_n=None, reverse=False):
        '''
        get best (or worst) rank
        :param metric: alpha_nDCG or ERR_IA
        :param top_n: number of top docs
        :param reverse: get the worst if True
        :return: a new DiversityQuery object with highest (or lowest) metric score
        '''
        top_n = top_n or len(self.docs)
        if metric == 'ERR_IA':
            score, ranks = best_ERR_IA(self._grade_matrix.T, reverse=reverse)
        elif metric == 'alpha_nDCG':
            score, ranks, score_list = best_alpha_nDCG(self._grade_matrix.T, alpha=self.ALPHA, reverse=reverse)
        else:
            raise Exception('unsupported metric {}'.format(metric))
        return DiversityQuery(self.query, self.qid, self.subtopics, self.docs[ranks][:top_n])

    def worst_rank(self, metric='alpha_nDCG', top_n=None):
        '''
        get worst rank
        :param metric: alpha_nDCG or ERR_IA
        :param top_n: number of top docs
        :return: a new DiversityQuery object with lowest metric score
        '''
        return self.best_rank(metric=metric, top_n=top_n, reverse=True)

    @staticmethod
    def get_pair_samples(best, metric, use_best_sample=True, perm_num=0):
        '''
        generate list-pairwise samples
        :param best: a DiversityQuery object with highest metric score
        :param metric: alpha_nDCG or ERR_IA
        :param use_best_sample: whether use best rank as context
        :param perm_num: number of negative context
        :return: list-pairwise samples
        '''
        result = []
        def get_pairs(dq, context, threshold=0):
            ind = [i for i in range(len(dq.docs)) if i not in context]
            metrics = [DiversityQuery(dq.query, dq.qid, dq.subtopics, dq.docs[context + [i]]).get_metric(metric=metric, normalize=False)
                       for i in range(len(dq.docs)) if i not in context]
            arg = np.argsort(metrics)
            pairs = []
            for i in range(len(metrics)):
                for j in range(i+1, len(metrics)):
                    if metrics[arg[j]] - metrics[arg[i]] > threshold:
                        pairs.append((ind[arg[i]], ind[arg[j]], metrics[arg[j]] - metrics[arg[i]]))
            return pairs
        def get_perm(dq, top, n):
            used = set()
            result = []
            while n > 0:
                perm = np.random.permutation(len(dq.docs))
                hash = '.'.join(str(perm[p]) for p in range(top))
                if hash not in used:
                    used.add(hash)
                    n -= 1
                    result.append(DiversityQuery(dq.query, dq.qid, dq.subtopics, dq.docs[perm]))
            return result
        for i in range(len(best.docs)):
            if use_best_sample:
                pairs = get_pairs(best, list(range(i)))
                if len(pairs) > 0:
                    result.append((best.docids[list(range(i))].tolist(), [(best.docids[p[0]], best.docids[p[1]], p[2]) for p in pairs]))
            if i > 0 and perm_num > 0:
                for dq in get_perm(best, top=i, n=min(perm_num, scipy.math.factorial(i))):
                    pairs = get_pairs(dq, list(range(i)))
                    if len(pairs) > 0:
                        result.append((dq.docids[list(range(i))], [(dq.docids[p[0]], dq.docids[p[1]], p[2]) for p in pairs]))
        return result