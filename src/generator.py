import tensorflow as tf
from src import config
from src import test
from src.SpectralGAN import data
class Generator(object):
    def __init__(self, n_node, n_layer):
        self.n_node = n_node

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.Variable(tf.random_normal([self.n_node, config.emb_dim],
                                                                 mean=0.01, stddev=0.02, dtype=tf.float32),
                                                                 name='features')

            self.weight_matrix = tf.Variable(tf.random_normal([n_layer, config.emb_dim, config.emb_dim],
                                                               mean=0.01, stddev=0.02, dtype=tf.float32),
                                                               name='weight')

        #self.adj_miss = tf.placeholder(tf.int32, shape=[n_node, n_node])
        self.eigen_vectors = tf.placeholder(tf.float32, shape=[self.n_node, config.n_eigs])
        self.eigen_values = tf.placeholder(tf.float32, shape=[config.n_eigs])
        self.node_id = tf.placeholder(tf.int32, shape=[config.missing_edge])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[config.missing_edge])
        self.reward = tf.placeholder(tf.float32, shape=[config.missing_edge])

        #adj_miss = tf.cast(self.adj_miss, tf.float32)
        #degree = tf.diag(tf.reciprocal(tf.reduce_sum(adj_miss, axis=1)))
        A_hat = tf.add(tf.matmul(self.eigen_vectors, tf.transpose(self.eigen_vectors)),
                       tf.matmul(self.eigen_vectors, tf.matmul(tf.diag(self.eigen_values),
                                                               tf.transpose(self.eigen_vectors))))
        all_embeddings = [self.embedding_matrix]
        for l in range(n_layer):
            weight_for_l = tf.gather(self.weight_matrix, l)
            embedding_matrix = tf.nn.sigmoid(tf.matmul(tf.matmul(A_hat,self.embedding_matrix),
                                                  weight_for_l))
            all_embeddings.append(embedding_matrix)

        all_embeddings = tf.concat(all_embeddings, 1)
        self.node_embedding = tf.nn.embedding_lookup(all_embeddings, self.node_id)  # batch_size * n_embed

        self.node_neighbor_embedding = tf.nn.embedding_lookup(all_embeddings,
                                                              self.node_neighbor_id)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1)
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)

        self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + config.lambda_gen * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding))

        user_embeddings, item_embeddings = tf.split(all_embeddings, [data.n_users, data.n_items])
        self.all_score = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(tf.reduce_mean(self.loss))
