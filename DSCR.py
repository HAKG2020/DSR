import tensorflow as tf
import os
import sys
import json
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from utility.helper import *
from utility.batch_test import *


class NGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj_S = data_config['norm_adj_S']
        self.norm_adj_R = data_config['norm_adj_R']
        self.n_nonzero_elems = self.norm_adj_R.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.transmat_R_dim = args.transmat_R_dim
        self.transmat_S_dim = args.transmat_S_dim
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)  # [64,64,64]
        self.n_layers = len(self.weight_size)  # 3
        self.weight_size_S = eval(args.layer_size_S)  # [64,64,64]
        self.n_layers_S = len(self.weight_size_S)  # 3

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()
        if self.alg_type in ['dscr']:
            self.ua_embeddings, self.us_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.u_s_embeddings = tf.nn.embedding_lookup(self.us_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings, self.u_s_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            # embedding initialization
            all_weights['user_embedding_S'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                          name='user_embedding_S')
            all_weights['user_embedding_R'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                          name='user_embedding_R')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            all_weights['transmat_R'] = tf.Variable(initializer([self.emb_dim * 4, 8]),
                                                    name='transmat_R')
            all_weights['transmat_S'] = tf.Variable(initializer([self.emb_dim * 3, 8]),
                                                    name='transmat_S')
            print('using xavier initialization,1-1-64')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        # self.weight_size_list = [self.emb_dim] + self.weight_size
        self.weight_size_list_R = [self.emb_dim] + self.weight_size
        self.weight_size_list_S = [self.emb_dim] + self.weight_size_S
        for k in range(self.n_layers):
            all_weights['W_gc_%d_R' % k] = tf.Variable(
                initializer([self.weight_size_list_R[k], self.weight_size_list_R[k + 1]]), name='W_gc_%d_R' % k)
            all_weights['b_gc_%d_R' % k] = tf.Variable(
                initializer([1, self.weight_size_list_R[k + 1]]), name='b_gc_%d_R' % k)

            all_weights['W_bi_%d_R' % k] = tf.Variable(
                initializer([self.weight_size_list_R[k], self.weight_size_list_R[k + 1]]), name='W_bi_%d_R' % k)
            all_weights['b_bi_%d_R' % k] = tf.Variable(
                initializer([1, self.weight_size_list_R[k + 1]]), name='b_bi_%d_R' % k)

            all_weights['W_mlp_%d_R' % k] = tf.Variable(
                initializer([self.weight_size_list_R[k], self.weight_size_list_R[k + 1]]), name='W_mlp_%d_R' % k)
            all_weights['b_mlp_%d_R' % k] = tf.Variable(
                initializer([1, self.weight_size_list_R[k + 1]]), name='b_mlp_%d_R' % k)
        for k in range(self.n_layers_S):
            all_weights['W_gc_%d_S' % k] = tf.Variable(
                initializer([self.weight_size_list_S[k], self.weight_size_list_S[k + 1]]), name='W_gc_%d_S' % k)
            all_weights['b_gc_%d_S' % k] = tf.Variable(
                initializer([1, self.weight_size_list_S[k + 1]]), name='b_gc_%d_S' % k)

            all_weights['W_bi_%d_S' % k] = tf.Variable(
                initializer([self.weight_size_list_S[k], self.weight_size_list_S[k + 1]]), name='W_bi_%d_S' % k)
            all_weights['b_bi_%d_S' % k] = tf.Variable(
                initializer([1, self.weight_size_list_S[k + 1]]), name='b_bi_%d_S' % k)

            all_weights['W_mlp_%d_S' % k] = tf.Variable(
                initializer([self.weight_size_list_S[k], self.weight_size_list_S[k + 1]]), name='W_mlp_%d_S' % k)
            all_weights['b_mlp_%d_S' % k] = tf.Variable(
                initializer([1, self.weight_size_list_S[k + 1]]), name='b_mlp_%d_S' % k)

        return all_weights

    def _split_A_hat_S(self, X):
        A_fold_hat = []

        # fold_len = (self.n_users + self.n_items) // self.n_fold
        fold_len = (self.n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout_S(self, X):
        A_fold_hat = []

        fold_len = (self.n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _split_A_hat_R(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout_R(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat_S = self._split_A_hat_node_dropout_S(self.norm_adj_S)
            A_fold_hat_R = self._split_A_hat_node_dropout_R(self.norm_adj_R)
        else:
            A_fold_hat_S = self._split_A_hat_S(self.norm_adj_S)
            A_fold_hat_R = self._split_A_hat_R(self.norm_adj_R)

        user_embeddings_R = self.weights['user_embedding_R']
        user_embeddings_S = self.weights['user_embedding_S']
        ego_embeddings_R = tf.concat([user_embeddings_R, self.weights['item_embedding']], axis=0)
        all_embeddings_R = [ego_embeddings_R]
        ego_embeddings_S = user_embeddings_S
        all_embeddings_S = [ego_embeddings_S]
        for k in range(0, self.n_layers_S):

            temp_embed_S = []
            for f in range(self.n_fold):
                temp_embed_S.append(tf.sparse_tensor_dense_matmul(A_fold_hat_S[f], ego_embeddings_S))

            # sum messages of neighbors.
            side_embeddings_S = tf.concat(temp_embed_S, 0)
            # transformed sum messages of neighbors.
            sum_embeddings_S = tf.nn.leaky_relu(
                tf.matmul(side_embeddings_S, self.weights['W_gc_%d_S' % k]) + self.weights['b_gc_%d_S' % k])

            # bi messages of neighbors.
            bi_embeddings_S = tf.multiply(ego_embeddings_S, side_embeddings_S)
            # transformed bi messages of neighbors.
            bi_embeddings_S = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_S, self.weights['W_bi_%d_S' % k]) + self.weights['b_bi_%d_S' % k])

            # non-linear activation.
            ego_embeddings_S = sum_embeddings_S + bi_embeddings_S

            # message dropout.
            ego_embeddings_S = tf.nn.dropout(ego_embeddings_S, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings_S = tf.math.l2_normalize(ego_embeddings_S, axis=1)

            all_embeddings_S += [norm_embeddings_S]
        for k in range(0, self.n_layers):

            temp_embed_R = []
            for f in range(self.n_fold):
                temp_embed_R.append(tf.sparse_tensor_dense_matmul(A_fold_hat_R[f], ego_embeddings_R))

            # sum messages of neighbors.
            side_embeddings_R = tf.concat(temp_embed_R, 0)
            # transformed sum messages of neighbors.
            sum_embeddings_R = tf.nn.leaky_relu(
                tf.matmul(side_embeddings_R, self.weights['W_gc_%d_R' % k]) + self.weights['b_gc_%d_R' % k])

            # bi messages of neighbors.
            bi_embeddings_R = tf.multiply(ego_embeddings_R, side_embeddings_R)
            # transformed bi messages of neighbors.
            bi_embeddings_R = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_R, self.weights['W_bi_%d_R' % k]) + self.weights['b_bi_%d_R' % k])

            # non-linear activation.
            ego_embeddings_R = sum_embeddings_R + bi_embeddings_R

            # message dropout.
            ego_embeddings_R = tf.nn.dropout(ego_embeddings_R, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings_R = tf.math.l2_normalize(ego_embeddings_R, axis=1)

            all_embeddings_R += [norm_embeddings_R]

        all_embeddings_R = tf.concat(all_embeddings_R, 1)
        all_embeddings_S = tf.concat(all_embeddings_S, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings_R, [self.n_users, self.n_items], 0)
        # print('u_g_embeddings',u_g_embeddings.shape)
        # print('i_g_embeddings',i_g_embeddings.shape)
        return u_g_embeddings, all_embeddings_S, i_g_embeddings

    def create_bpr_loss(self, users, u_s_embeddings, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        difference = tf.matmul(users, self.weights['transmat_R']) - tf.matmul(u_s_embeddings,
                                                                              self.weights['transmat_S'])
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items) + tf.nn.l2_loss(difference) #+ tf.nn.l2_loss(u_s_embeddings) 
        regularizer = regularizer / self.batch_size
        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
        variable_name = [v.name for v in tf.trainable_variables()]
        print(variable_name)
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj_R, norm_adj_R, mean_adj_R, plain_adj_S, norm_adj_S, mean_adj_S = data_generator.get_adj_mat()

    if args.adj_type == 'plain':
        config['norm_adj_S'] = plain_adj_S
        config['norm_adj_R'] = plain_adj_R
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj_R'] = norm_adj_R
        config['norm_adj_S'] = norm_adj_S
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['norm_adj_R'] = mean_adj_R
        config['norm_adj_S'] = mean_adj_S
        print('use the gcmc adjacency matrix')

    else:
        config['norm_adj_R'] = mean_adj_R + sp.eye(mean_adj_R.shape[0])
        config['norm_adj_S'] = mean_adj_S + sp.eye(mean_adj_S.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = NGCF(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = '0,3'
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    gpu_options = tf.GPUOptions(allow_growth=True)
    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            ####################################################################
            variable_name = [v.name for v in tf.trainable_variables()]
            print(variable_name)
            item_emb_matrix = tf.get_default_graph().get_tensor_by_name('item_embedding:0')
            user_embR_matrix = tf.get_default_graph().get_tensor_by_name('user_embedding_R:0')
            ####################################################################
            #print('rating_matrix_shape', plain_adj_R.shape)
            print('item_matrix_shape', item_emb_matrix.shape)
            u_idlist = [1308, 3091, 4810, 158, 4974, 3983]
            emb_dict = {}
            user_emb_dict = {}
            # adj_R = plain_adj_R.toarray()
            with open('plain_adj_R.pickle','wb') as file:
                pickle.dump(plain_adj_R,file)
            for u_id in u_idlist:
                row = plain_adj_R[u_id].toarray()
                index = np.nonzero(row)[1]
                print(u_id, index)
                count = 0
                for id in index:
                    index[count] = id - 7376
                    count +=1
                print(u_id, index)
                item_embeds = tf.nn.embedding_lookup(item_emb_matrix, index).eval(session = sess)
                user_emb= tf.nn.embedding_lookup(user_embR_matrix, np.array([u_id])).eval(session = sess)
                emb_dict.update({u_id:item_embeds})
                user_emb_dict.update({u_id: user_emb})
            ##################################################################
            with open('embeddings.pickle','wb') as file:
                pickle.dump(emb_dict,file)
                pickle.dump(user_emb_dict, file)
            print('load the pretrained model parameters from: ', pretrain_path)


            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['ndcg'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, regs=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.regs, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            # print('users, pos_items, neg_items', len(users), len(pos_items), len(neg_items))
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                feed_dict={model.users: users, model.pos_items: pos_items,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout),
                           model.neg_items: neg_items})
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['ndcg'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
