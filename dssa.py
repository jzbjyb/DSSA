# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from sklearn.base import BaseEstimator, ClassifierMixin
import time
from type import DiversityQuery

def select_by_index(value, index, zero_base=True):
    select_dim = tf.shape(value)[1]
    mask = tf.equal(tf.range(select_dim), tf.expand_dims(index, axis=-1) - (0 if zero_base else 1))
    return tf.boolean_mask(value, mask)

def tensormul(tensor, arrs):
    dims = [d.value for d in tensor.get_shape()]
    batch = tf.shape(arrs[0])[0]
    ultimate = [batch] + dims
    tensor = tf.reshape(tf.tile(tensor, [batch] + [1] * (len(dims) - 1)), ultimate)
    arrs = [tf.reshape(tf.tile(tf.reshape(arrs[i], [batch, dims[i], 1]),
                               [1, int(np.product(dims[:i])), int(np.product(dims[i+1:]))]),
                       ultimate) for i in range(len(arrs))]
    for arr in arrs:
        tensor *= arr
    return tf.reduce_sum(tensor, list(range(1, len(ultimate))))

def interaction_op(emb1, emb2, var_name, op='general'):
    s1 = emb1.get_shape()[1].value
    s2 = emb2.get_shape()[1].value
    if op == 'general':
        variable = vs.get_variable(var_name, [s1, s2])
        return tensormul(variable, [emb1, emb2])
    elif op == 'dot':
        return tf.reduce_sum(emb1 * emb2, reduction_indices=1)

class RNNCellBase(core_rnn_cell.RNNCell):
    def zero_state_np(self, batch_size, dtype):
        states = np.zeros((batch_size, self.state_size), dtype=dtype)
        return states

class DSSACell(RNNCellBase):
    '''
    DSSA cell: combination of traditional RNN and pooling
    '''
    VALID_OP = {'general', 'dot'}
    VALID_POOL = {'max', 'min'}

    def __init__(self, cell, n_doc_emb, n_rel_feat, n_query_emb, most_n_subquery,
                 lambdaa=0.5, op='general', pool='max', state_is_tuple=False):
        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError('cell should be instance of RNNCell')
        if op not in DSSACell.VALID_OP:
            raise ValueError('op not valid, it should be one of {}'
                             .format(', '.join(map(lambda x: '"' + x + '"', DSSACell.VALID_OP))))
        if pool not in DSSACell.VALID_POOL:
            raise ValueError('pool not valid, it should be one of {}'
                             .format(', '.join(map(lambda x: '"' + x + '"', DSSACell.VALID_POOL))))
        if nest.is_sequence(cell.state_size) != state_is_tuple:
            raise ValueError('base cell state_is_tuple is not consistent with the current. base state size is: {}'.format(cell.state_size))
        self.cell = cell
        self.n_doc_emb = n_doc_emb
        self.n_rel_feat = n_rel_feat
        self.n_query_emb = n_query_emb
        self.most_n_subquery = most_n_subquery
        self.lambdaa = lambdaa
        self.op = op
        self.pool = pool
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        ss = (self.cell.state_size, self.most_n_subquery, self.most_n_subquery)
        if self.state_is_tuple:
            return ss
        return sum(ss)

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope('Transformation'):
                batch_size = tf.shape(inputs)[0]
                input_dim = tf.shape(inputs)[1]
                # the samples in the same batch must have same number of subtopics
                to_attn_len = tf.to_int32(tf.reduce_max(array_ops.slice(inputs, [0, input_dim - 1], [-1, -1]))) - 1
                to_attn_len = tf.maximum(to_attn_len, 0) # avoid bug for paddings
                query_rel = array_ops.slice(inputs, [0, 0], [-1, self.n_rel_feat])
                subquery_rel = tf.reshape(array_ops.slice(inputs, [0, self.n_rel_feat], [-1, self.n_rel_feat * to_attn_len]),
                                          [batch_size, to_attn_len, self.n_rel_feat])
                doc_emb = array_ops.slice(inputs, [0, self.n_rel_feat * self.most_n_subquery], [-1, self.n_doc_emb])
                query_emb = array_ops.slice(inputs, [0, self.n_rel_feat * self.most_n_subquery + self.n_doc_emb], [-1, self.n_query_emb])
                to_attn = tf.reshape(array_ops.slice(inputs, [0, self.n_rel_feat * self.most_n_subquery + self.n_doc_emb + self.n_query_emb + 1],
                                                     [-1, (self.n_query_emb + 1) * to_attn_len]),
                                     [batch_size, to_attn_len, self.n_query_emb + 1])
                initial_weight = to_attn[:, :, self.n_query_emb]
                subquery_emb = to_attn[:, :, :self.n_query_emb]
                last_state = state[:, :self.cell.state_size]
                last_attn_rnn = array_ops.slice(state, [0, self.cell.state_size], [-1, to_attn_len])
                last_attn_pool = array_ops.slice(state, [0, self.cell.state_size + self.most_n_subquery], [-1, to_attn_len])
                last_attn = last_attn_rnn + last_attn_pool
                last_attn = tf.nn.softmax(last_attn + tf.log(initial_weight))
            with vs.variable_scope('DocumentSequenceRepresentation'):
                cur_output, cur_state = self.cell(doc_emb, last_state)
            with vs.variable_scope('Scoring'):
                lambda_factor = vs.get_variable('Lambda', [1], initializer=tf.constant_initializer(self.lambdaa, dtype=tf.float32), trainable=False)
                # diversity score
                div_score = tf.zeros([batch_size])
                relw = vs.get_variable('RelW', [self.n_rel_feat, 1])
                sub_rel = tf.reduce_sum(subquery_rel * tf.reshape(relw, [self.n_rel_feat]), reduction_indices=2)
                div_score += tf.reduce_sum(sub_rel * last_attn, reduction_indices=1)
                div_score += interaction_op(doc_emb, tf.reduce_sum(tf.expand_dims(last_attn, -1) * subquery_emb, reduction_indices=1), 'OutW', op=self.op)
                # relevance score
                rel_score = tf.zeros([batch_size])
                rel_score += tf.reshape(tf.matmul(query_rel, relw), [batch_size])
                vs.get_variable_scope().reuse_variables()
                rel_score += interaction_op(doc_emb, query_emb, 'OutW', op=self.op)
                score = lambda_factor * div_score + (1.0 - lambda_factor) * rel_score
            with vs.variable_scope('SubtopicAttention'):
                # rnn attention part
                attn_rnn = tf.reshape(interaction_op(tf.reshape(tf.tile(cur_output, [1, to_attn_len]), [-1, self.cell.output_size]),
                                                     tf.reshape(subquery_emb, [-1, self.n_query_emb]), 'AttnW', op=self.op),
                                      [batch_size, to_attn_len])
                # pooling attention part
                attn_relw = vs.get_variable('AttnRelW', [self.n_rel_feat, 1])
                attn_sub_rel = tf.reduce_sum(subquery_rel * tf.reshape(attn_relw, [self.n_rel_feat]), reduction_indices=2)
                if self.pool == 'max':
                    attn_pool = tf.reduce_max(tf.stack([attn_sub_rel, last_attn_pool], axis=2), reduction_indices=2)
                elif self.pool == 'min':
                    attn_pool = tf.reduce_min(tf.stack([attn_sub_rel, last_attn_pool], axis=2), reduction_indices=2)
            score = tf.reshape(score, [batch_size, 1])
            next_state = (cur_state,
                          tf.concat([attn_rnn, tf.zeros((batch_size, self.most_n_subquery - to_attn_len))], axis=1),
                          tf.concat([attn_pool, tf.zeros((batch_size, self.most_n_subquery - to_attn_len))], axis=1))
            if self.state_is_tuple:
                return score, next_state
            return score, tf.concat(list(next_state), axis=1)

    def zero_state_np(self, batch_size, dtype):
        if self.pool == 'max':
            v = np.finfo(np.float32).min
        elif self.pool == 'min':
            v = np.finfo(np.float32).max
        if self.state_is_tuple:
            state_base = tuple(np.zeros([batch_size, d]) for d in self.cell.state_size)
        else:
            state_base = np.zeros([batch_size, self.cell.state_size])
        state_rnn = np.zeros((batch_size, self.most_n_subquery), dtype=dtype)
        state_pool = np.ones((batch_size, self.most_n_subquery), dtype=dtype) * v
        if self.state_is_tuple:
            return (state_base, state_rnn, state_pool)
        return np.concatenate([state_base, state_rnn, state_pool], axis=1)

    def zero_state(self, batch_size, dtype):
        if self.pool == 'max':
            v = tf.float32.min
        elif self.pool == 'min':
            v = tf.float32.max
        state_base = self.cell.zero_state(batch_size, dtype)
        state_rnn = array_ops.zeros([batch_size, self.most_n_subquery], dtype=dtype)
        state_pool = tf.fill([batch_size, self.most_n_subquery], v)
        if self.state_is_tuple:
            return (state_base, state_rnn, state_pool)
        return tf.concat([state_base, state_rnn, state_pool], axis=1)

class DSSA(BaseEstimator, ClassifierMixin):
    '''
    DSSA model
    '''
    VALID_INTERACTION = {'general', 'dot'}
    VALID_CELL_TYPE = {'vanilla', 'LSTM', 'GRU'}
    VALID_OPTIMIZATION = {'listpair'}

    def __init__(self,
                 n_rel_feat=10,
                 n_doc_emb=10,
                 n_query_emb=10,
                 hidden_size=10,
                 interaction='general',
                 cell_type='vanilla',
                 lambdaa=0.5,
                 most_n_subquery=10,
                 most_n_pair=10,
                 most_n_doc=10,
                 doc_emb=np.ones((10, 110)),
                 query_emb=np.ones((10, 11)),
                 learning_rate=.1,
                 n_epochs=10,
                 batch_size=10,
                 optimization='listpair',
                 save_epochs=None,
                 verbose=True,
                 random_seed=0,
                 reuse_model=None,
                 save_model=None):
        self.n_rel_feat = n_rel_feat
        self.n_doc_emb = n_doc_emb
        self.n_query_emb = n_query_emb
        self.hidden_size = hidden_size
        self.interaction = interaction
        self.cell_type = cell_type
        self.lambdaa = lambdaa
        self.most_n_subquery = most_n_subquery
        self.most_n_pair = most_n_pair
        self.most_n_doc = most_n_doc
        self.doc_emb = doc_emb
        self.query_emb = query_emb
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimization = optimization
        self.save_epochs = save_epochs
        self.verbose = verbose
        self.random_seed = random_seed
        self.reuse_model = reuse_model
        self.save_model = save_model

    def check_params(self):
        if self.interaction not in DSSA.VALID_INTERACTION:
            raise ValueError('interaction not valid, it should be one of {}'
                             .format(', '.join(map(lambda x: '"' + x + '"', DSSA.VALID_INTERACTION))))
        if self.cell_type not in DSSA.VALID_CELL_TYPE:
            raise ValueError('cell_type not valid, it should be one of {}'
                             .format(', '.join(map(lambda x: '"' + x + '"', DSSA.VALID_CELL_TYPE))))
        if not isinstance(self.doc_emb, np.ndarray) or not isinstance(self.query_emb, np.ndarray):
            raise ValueError('both doc_emb and query_emb should by instance of numpy.ndarray')
        self.doc_emb_actual_size = self.n_rel_feat * self.most_n_subquery + self.n_doc_emb
        if self.doc_emb.shape[1] != self.doc_emb_actual_size:
            raise ValueError('doc_emb shape[1] is unexpected. {} is desired while we got {}'
                             .format(self.doc_emb_actual_size, self.doc_emb.shape[1]))
        self.query_emb_actual_size = self.n_query_emb + 1
        if self.query_emb.shape[1] != self.query_emb_actual_size:
            raise ValueError('query_emb shape[1] is unexpected. {} is desired while we got {}'
                             .format(self.query_emb_actual_size, self.query_emb.shape[1]))
        if self.optimization not in DSSA.VALID_OPTIMIZATION:
            raise ValueError('optimization not valid, it should be one of {}'
                             .format(', '.join(map(lambda x: '"' + x + '"', DSSA.VALID_OPTIMIZATION))))
        self.input_dim = 1 + self.most_n_subquery
        self.expand_input_dim = self.n_rel_feat * self.most_n_subquery + self.n_doc_emb + (self.n_query_emb + 1) * self.most_n_subquery
        if self.reuse_model and not hasattr(self, 'session_'): # read model from file
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self.random_seed)
                with vs.variable_scope('DSSA', initializer=tf.uniform_unit_scaling_initializer(seed=self.random_seed)) as scope:
                    self.build_graph()
                    scope.reuse_variables()
                    self.build_graph_test()
            self.session_ = tf.Session(graph=self.graph_)
            print('load model from "{}"'.format(self.reuse_model))
            self.saver.restore(self.session_, self.reuse_model)

    @staticmethod
    def get_cell(cell_type, hidden_size):
        if cell_type == 'vanilla':
            return core_rnn_cell.BasicRNNCell(num_units=hidden_size)
        elif cell_type == 'GRU':
            return core_rnn_cell.GRUCell(num_units=hidden_size)
        elif cell_type == 'LSTM':
            return core_rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=False)

    @property
    def weights(self):
        if not hasattr(self, 'session_'):
            raise AttributeError('need fit or fit_iterable to be called before getting weights')
        return [(v.name, v.eval(self.session_)) for v in self.graph_.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DSSA')]

    def batcher(self, X, y=None):
        for q in X:
            n_sample = q['train_x'].shape[0]
            if n_sample <= 0:
                continue
            for i in range(0, n_sample, self.batch_size):
                yield q['train_x'][i:i + self.batch_size], None

    def batch_to_feeddict(self, X, y):
        if self.optimization == 'listpair':
            return self.batch_to_feeddict_listpair(X, y)

    def batch_to_feeddict_listpair(self, X, y):
        batch_num = X.shape[0]
        actual_dim = self.most_n_doc * (self.input_dim + 1) + self.most_n_pair * (2 * (self.input_dim + 1) + 1 + 1) + 1 + 1
        if X.shape[1] != actual_dim:
            raise ValueError('input X\'s dimension is unexpected. {} is desired while we got {}'.format(actual_dim, X.shape[1]))
        # assemble feed_dict
        feed_dict = {}
        feed_dict[self.context_input] = X[:, :self.most_n_doc * (self.input_dim + 1)].reshape(batch_num, self.most_n_doc, self.input_dim + 1)
        feed_dict[self.context_input_len] = X[:, -2]
        feed_dict[self.pair_input_len] = X[:, -1]
        pair = X[:, self.most_n_doc * (self.input_dim + 1):-2].reshape(batch_num, self.most_n_pair, -1)
        feed_dict[self.pair_input] = pair[:, :, :-2].reshape(batch_num, self.most_n_pair, 2, self.input_dim + 1)
        feed_dict[self.pair_target] = pair[:, :, -2].astype(np.int32)
        feed_dict[self.pair_weight] = pair[:, :, -1]
        return feed_dict

    def build_graph_test(self):
        if not hasattr(self, 'cell'):
            raise AttributeError('need build train graph before build test graph')
        with vs.variable_scope('TestInput'):
            self.test_input = tf.placeholder(tf.float32, shape=[None, None, self.input_dim + 1], name='test_input')
            self.test_input_len = tf.placeholder(tf.int32, shape=[None], name='test_input_len')
            self.test_initial_state = tf.placeholder(tf.float32, shape=[None, self.cell.state_size], name='test_initial_state')
            dim = tf.shape(self.test_input)
            batch_num = dim[0]
            doc_num = dim[1]
        with vs.variable_scope('EmbeddingLookup'):
            tid = tf.nn.embedding_lookup(self.doc_emb_W, tf.to_int32(self.test_input[:, :, :1]))
            tiq = tf.nn.embedding_lookup(self.query_emb_W, tf.to_int32(self.test_input[:, :, 1:1 + self.most_n_subquery]))
            test_input_expand = tf.concat([tf.reshape(tid, [batch_num, doc_num, -1]),
                                           tf.reshape(tiq, [batch_num, doc_num, -1]),
                                           self.test_input[:, :, self.input_dim:]], axis=2)
            test_input_expand.set_shape([None, None, self.expand_input_dim + 1])
        with vs.variable_scope('DSSA-RNNMP'):
            outputs, self.states = tf.nn.dynamic_rnn(
                initial_state=self.test_initial_state,
                cell=self.cell,
                dtype=tf.float32,
                sequence_length=self.test_input_len,
                inputs=test_input_expand)
            self.outputs_test = tf.reshape(select_by_index(outputs, self.test_input_len, zero_base=False), [batch_num])

    def build_graph(self):
        if self.optimization == 'listpair':
            self.build_graph_listpair()

    def build_graph_listpair(self):
        with vs.variable_scope('TrainInput'):
            self.context_input = tf.placeholder(tf.float32, shape=[None, self.most_n_doc, self.input_dim + 1], name='context_input')
            self.context_input_len = tf.placeholder(tf.int32, shape=[None], name='context_input_len')
            self.pair_input = tf.placeholder(tf.float32, shape=[None, self.most_n_pair, 2, self.input_dim + 1], name='pair_input_raw')
            self.pair_weight = tf.placeholder(tf.float32, shape=[None, self.most_n_pair], name='pair_weight_raw')
            self.pair_target = tf.placeholder(tf.int32, shape=[None, self.most_n_pair], name='pair_target_raw')
            self.pair_input_len = tf.placeholder(tf.int32, shape=[None], name='pair_input_len')
        with vs.variable_scope('TrainInputShape'):
            batch_num = tf.shape(self.context_input)[0]
        with vs.variable_scope('EmbeddingLookup'):
            self.doc_emb_W = tf.constant(self.doc_emb, dtype=tf.float32, name='doc_emb_W')
            self.query_emb_W = tf.constant(self.query_emb, dtype=tf.float32, name='query_emb_W')
            # look up context_input doc & query embedding
            cid = tf.nn.embedding_lookup(self.doc_emb_W, tf.to_int32(self.context_input[:, :, :1]))
            ciq = tf.nn.embedding_lookup(self.query_emb_W, tf.to_int32(self.context_input[:, :, 1:1 + self.most_n_subquery]))
            context_input_expand = tf.concat([tf.reshape(cid, [batch_num, self.most_n_doc, -1]),
                                              tf.reshape(ciq, [batch_num, self.most_n_doc, -1]),
                                              self.context_input[:, :, self.input_dim:]], axis=2)
            context_input_expand.set_shape([None, self.most_n_doc, self.expand_input_dim + 1])
            # look up pair_input doc & query embedding
            pid = tf.nn.embedding_lookup(self.doc_emb_W, tf.to_int32(self.pair_input[:, :, :, :1]))
            piq = tf.nn.embedding_lookup(self.query_emb_W, tf.to_int32(self.pair_input[:, :, :, 1:1 + self.most_n_subquery]))
            pair_input_expand = tf.concat([tf.reshape(pid, [batch_num, self.most_n_pair, 2, -1]),
                                           tf.reshape(piq, [batch_num, self.most_n_pair, 2, -1]),
                                           self.pair_input[:, :, :, self.input_dim:]], axis=3)
            pair_input_expand.set_shape([None, self.most_n_pair, 2, self.expand_input_dim + 1])
        with vs.variable_scope('DSSA-RNNMP'):
            cell = DSSA.get_cell(self.cell_type, self.hidden_size)
            self.cell = DSSACell(cell=cell, n_doc_emb=self.n_doc_emb, n_rel_feat=self.n_rel_feat,
                                 n_query_emb=self.n_query_emb, most_n_subquery=self.most_n_subquery,
                                 lambdaa=self.lambdaa, op=self.interaction, pool='max', state_is_tuple=False)
            outputs, context_states = tf.nn.dynamic_rnn(
                cell=self.cell,
                dtype=tf.float32,
                sequence_length=self.context_input_len,
                inputs=context_input_expand
            )
            with vs.variable_scope('ContextPairTransformation'):
                context_states = tf.reshape(tf.tile(context_states, [1, self.most_n_pair * 2]),
                                            [batch_num, self.most_n_pair, 2, self.cell.state_size])
                pair_mask = tf.range(self.most_n_pair) < tf.expand_dims(self.pair_input_len, -1)
                pair_weight_selected = tf.boolean_mask(self.pair_weight, pair_mask, name='pair_weight_select')
                pair_target_selected = tf.boolean_mask(self.pair_target, pair_mask, name='pair_target_select')
                pair_input_selected = tf.boolean_mask(pair_input_expand, pair_mask, name='pair_input_select')
                self.pair_input_selected = tf.reshape(pair_input_selected, [-1, 1, self.expand_input_dim + 1])
                context_states_selected = tf.boolean_mask(context_states, pair_mask, name='pair_states_select')
                self.context_states_selected = tf.reshape(context_states_selected, [-1, self.cell.state_size])
                actual_batch_num = tf.shape(pair_weight_selected)[0]
            vs.get_variable_scope().reuse_variables()
            outputs, second_states = tf.nn.dynamic_rnn(
                initial_state=self.context_states_selected,
                cell=self.cell,
                dtype=tf.float32,
                sequence_length=tf.fill([actual_batch_num * 2], 1),
                inputs=self.pair_input_selected
            )
        with vs.variable_scope('Loss'):
            self.outputs = tf.reshape(outputs, [actual_batch_num, 2])
            self.prob = tf.nn.softmax(self.outputs, name='pair_prob')
            target = tf.reshape(select_by_index(self.prob, pair_target_selected, zero_base=True), [actual_batch_num])
            self.accuracy = tf.reduce_sum(tf.cast(target > 0.5, tf.float32)) /\
                            tf.cast(tf.reduce_prod(tf.shape(target)), tf.float32)
            self.log_loss = -tf.reduce_sum(pair_weight_selected * tf.log(target), name='log_loss')
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.log_loss)
        self.init_all_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def fit_iterable(self, X, y=None):
        # check params
        self.check_params()
        # init graph and session
        if not hasattr(self, 'session_'):
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self.random_seed)
                with vs.variable_scope('DSSA', initializer=tf.uniform_unit_scaling_initializer(seed=self.random_seed)) as scope:
                    self.build_graph()
                    scope.reuse_variables()
                    self.build_graph_test()
            self.session_ = tf.Session(graph=self.graph_)
            self.session_.run(self.init_all_vars)
        # iteration
        for epoch in range(self.n_epochs):
            epoch += 1
            start = time.time()
            perm = np.random.permutation(len(X))
            loss_list, accuracy_list = [], []
            for i, (bX, by) in enumerate(self.batcher(X[perm], y[perm] if isinstance(y, np.ndarray) else None)):
                fd = self.batch_to_feeddict(bX, by)
                if self.optimization == 'listpair':
                    log_loss, accuracy, _ = \
                        self.session_.run([self.log_loss, self.accuracy, self.trainer], feed_dict=fd)
                    accuracy_list.append(accuracy)
                    loss_list.append(log_loss)
            if self.verbose: # output epoch stat
                print('{:<10}\t{:>7}:{:>5.3f}:{:>7.3f}'
                      .format('EPO[{}_{:>3.1f}]'.format(epoch, (time.time() - start)/60), 'train', np.mean(accuracy_list), np.mean(loss_list)), end='')
            if self.save_epochs and epoch % self.save_epochs == 0: # save the model
                if self.save_model:
                    self.saver.save(self.session_, self.save_model)
                yield
            elif epoch == self.n_epochs and self.save_epochs and self.n_epochs % self.save_epochs != 0:  # save the final model
                if self.save_model:
                    self.saver.save(self.session_, self.save_model)
                yield
            if self.verbose:
                print('')

    def fit(self, X, y=None):
        list(self.fit_iterable(X, y))
        return self

    def decision_function(self, X):
        self.check_params()
        if not hasattr(self, 'session_'):
            raise AttributeError('need fit or fit_iterable to be called before getting weights')
        ranks = []
        for q in X:
            select = []
            raw_data = q['test_x']
            n_doc = raw_data.shape[0]
            for r in range(n_doc):
                remain_ind = [i for i in range(n_doc) if i not in select]
                batch_input = np.array([[raw_data[i]] for i in remain_ind])
                out, states = self.session_.run([self.outputs_test, self.states], feed_dict={
                    self.test_input: batch_input,
                    self.test_input_len: np.ones((len(remain_ind))),
                    self.test_initial_state:
                        np.tile(last_states, [len(remain_ind)]).reshape(len(remain_ind), self.cell.state_size) if r > 0
                        else self.cell.zero_state_np(len(remain_ind), np.float32)
                })
                next = np.argmax(out)
                select.append(remain_ind[next])
                last_states = states[next]
            ranks.append(select)
        return ranks

    def predict(self, X):
        return self.decision_function(X)

    def score(self, X, y, sample_weight=None):
        ranks = self.predict(X)
        metrics = []
        for i in range(len(X)):
            dq = y[i]
            ndq = DiversityQuery(dq.query, dq.qid, dq.subtopics, dq.docs[ranks[i]])
            metrics.append(ndq.get_metric('alpha_nDCG'))
        return np.mean(metrics)

    def __del__(self):
        if hasattr(self, 'session_'):
            self.session_.close()