import tensorflow as tf
from src import config

class Discriminator(object):
    def __init__(self, data, n_node, n_layer):
        self.n_node = n_node
        self.data = data

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.Variable(tf.random_normal([self.n_node, config.emb_dim],
                                                                 mean=0.01, stddev=0.02, dtype=tf.float32),
                                                                 name='features')

            self.weight_matrix = tf.Variable(tf.random_normal([n_layer, config.emb_dim, config.emb_dim],
                                                               mean=0.01, stddev=0.02, dtype=tf.float32),
                                                               name='weight')
            self.bias = tf.Variable(tf.random_normal([n_layer, config.emb_dim],
                                                     mean=0.01, stddev=0.02, dtype=tf.float32),
                                    name='bias')

        #self.adj_miss = tf.placeholder(tf.int32, shape=[n_node, n_node])
        self.eigen_vectors = tf.placeholder(tf.float32, shape=[self.n_node, config.n_eigs])
        self.eigen_values = tf.placeholder(tf.float32, shape=[config.n_eigs])
        self.node_id = tf.placeholder(tf.int32, shape=[config.missing_edge*2])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[config.missing_edge*2])
        self.label = tf.placeholder(tf.float32, shape=[config.missing_edge*2])

        A_hat = tf.add(tf.matmul(self.eigen_vectors, tf.transpose(self.eigen_vectors)),
                       tf.matmul(self.eigen_vectors, tf.matmul(tf.diag(self.eigen_values),
                                                               tf.transpose(self.eigen_vectors))))

        all_embeddings = [self.embedding_matrix]
        for l in range(n_layer):
            weight_for_l = tf.gather(self.weight_matrix, l)
            bias_for_l = tf.gather(self.bias, l)
            embedding_matrix = tf.nn.sigmoid(tf.add(tf.matmul(tf.matmul(A_hat,self.embedding_matrix),
                                                            weight_for_l), bias_for_l))
            all_embeddings.append(embedding_matrix)

        self.all_embeddings = tf.concat(all_embeddings, 1)
        self.node_embedding = tf.nn.embedding_lookup(self.all_embeddings, self.node_id)
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.all_embeddings,
                                                              self.node_neighbor_id)
        self.score = tf.reduce_sum(tf.multiply(self.node_embedding, self.node_neighbor_embedding), axis=1)


        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) \
                + config.lambda_dis * ( tf.nn.l2_loss(self.node_neighbor_embedding) +
                                                tf.nn.l2_loss(self.node_embedding))

        user_embeddings, item_embeddings = tf.split(self.all_embeddings, [self.data.n_users, self.data.n_items])
        self.all_score = tf.matmul(user_embeddings, item_embeddings, transpose_b=True)

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(tf.reduce_mean(self.loss))
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))
