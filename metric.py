# -*- coding:utf-8 -*-

import numpy as np
from functools import reduce

def div_metric_param(grade_list, subtopics=None, k=20, grade_max=None):
    '''
    parameter check function
    :param grade_list: a np.ndarry with shape [#subtopic, #doc]
    :param subtopics: subtopic weights
    :param k: evaluation based on top k docs
    :param grade_max: maximum grade
    :return: checked parameters
    '''
    grade_max = grade_max or np.max(grade_list)
    if type(grade_list) is not np.ndarray:
        grade_list = np.array(grade_list)
    if len(grade_list.shape) == 1:
        grade_list = grade_list.reshape((1, len(grade_list)))
    n_subtopics = grade_list.shape[0]
    n_docs = grade_list.shape[1]
    if subtopics is None:
        subtopics = np.ones(n_subtopics) / n_subtopics
    elif type(subtopics) is not np.ndarray:
        subtopics = np.array(subtopics)
    k = k or n_docs
    if k > n_docs:
        raise ValueError('metric k overflow, max value is {} while you access {}'.format(n_docs, k))
    if subtopics.shape[0] != n_subtopics:
        raise ValueError('subtopic dimension incompatible')
    return grade_list, subtopics, k, grade_max

def best_ERR_IA(grade_list, subtopics=None, k=None, grade_max=None, reverse=False):
    '''
    calculate best ERR_IA score of current docs
    :param grade_list: a np.ndarry with shape [#subtopic, #doc]
    :param subtopics: subtopic weights
    :param k: evaluation based on top k docs
    :param grade_max: maximum grade
    :param reverse: get the worst if True
    :return: score, rank
    '''
    grade_list, subtopics, k, grade_max = div_metric_param(grade_list, subtopics, k, grade_max)
    grade_list = (np.power(2, grade_list) - 1) / np.power(2, grade_max)
    n_subtopics, n_docs = grade_list.shape[0], grade_list.shape[1]
    mask = np.zeros(n_docs)
    discount = np.ones(n_subtopics)
    discount *= subtopics
    score, rank = 0, []
    for i in range(k):
        scores = np.matmul(grade_list.T, discount) + mask
        r = np.argmax(scores) if not reverse else np.argmin(scores)
        discount *= 1 - grade_list[:,r]
        score += scores[r]/(i+1)
        rank.append(r)
        mask[r] = np.finfo(np.float32).min if not reverse else np.finfo(np.float32).max
    return score, rank

def ERR_IA(grade_list, subtopics=None, k=None, grade_max=None):
    '''
    calculate ERR_IA score of current docs
    :param grade_list: a np.ndarry with shape [#subtopic, #doc]
    :param subtopics: subtopic weights
    :param k: evaluation based on top k docs
    :param grade_max: maximum grade
    :return: score
    '''
    grade_list, subtopics, k, grade_max = div_metric_param(grade_list, subtopics, k, grade_max)
    grade_list = (np.power(2, grade_list) - 1) / np.power(2, grade_max)
    return np.dot([reduce(lambda t,i: [t[0]*(1-topic[i]), t[1]+t[0]*topic[i]/(i+1)] if i<k else t,
                          range(len(topic)), [1,0])[1] for topic in grade_list], subtopics)

def best_alpha_nDCG(grade_list, alpha=0.5, subtopics=None, k=None, grade_max=None, reverse=False):
    '''
    calculate best alpha_nDCG score of current docs
    :param grade_list: a np.ndarry with shape [#subtopic, #doc]
    :param alpha: alpha
    :param subtopics: subtopic weights
    :param k: evaluation based on top k docs
    :param grade_max: maximum grade
    :param reverse: get the worst if True
    :return: score, rank, scores for each step
    '''
    grade_list, subtopics, k, grade_max = div_metric_param(grade_list, subtopics, k, grade_max)
    alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
    grade_list = (grade_list > 0).astype(int)
    mask = np.zeros(n_docs)
    discount = np.zeros(n_subtopics)
    score, rank, score_list = 0, [], []
    for i in range(k):
        scores = np.matmul(grade_list.T, np.power(alpha, discount)) + mask
        r = np.argmax(scores) if not reverse else np.argmin(scores)
        discount += grade_list[:,r]
        score += scores[r]/np.log2(i+2)
        score_list.append(score)
        rank.append(r)
        mask[r] = np.finfo(np.float32).min if not reverse else np.finfo(np.float32).max
    return score, rank, score_list

def alpha_nDCG(grade_list, alpha=0.5, subtopics=None, k=None, grade_max=None, normalization=None):
    '''
    calculate alpha_nDCG score of current docs
    :param grade_list: a np.ndarry with shape [#subtopic, #doc]
    :param alpha: alpha
    :param subtopics: subtopic weights
    :param k: evaluation based on top k docs
    :param grade_max: maximum grade
    :param normalization: normalization
    :return: score
    '''
    grade_list, subtopics, k, grade_max = div_metric_param(grade_list, subtopics, k, grade_max)
    alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
    grade_list = (grade_list>0).astype(int)
    cum = reduce(lambda t,i:  [t[0]+(grade_list[:,i]),
                               t[1]+np.sum(np.dot(np.power(alpha,t[0]), grade_list[:,i]))/np.log2(i+2)],
                 range(k), [np.zeros(n_subtopics), 0])[1]
    if normalization is not None and normalization > 0:
        cum /= normalization
    return cum