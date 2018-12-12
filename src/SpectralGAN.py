import os
import numpy as np
import tensorflow as tf
from src import config
from src import generator
from src import discriminator
import random
from src import load_data
from src import utils
from src import test
from scipy.sparse import linalg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"]= '0'
data = load_data.Data(train_file=config.train_filename, test_file=config.test_filename)


class SpectralGAN(object):
    def __init__(self, n_users, n_items, R):
        print("reading graphs...")
        self.n_users, self.n_items = n_users, n_items
        self.n_node = n_users + n_items
        # construct graph
        self.R = R
        self.eigenvalues, self.eigenvectors = linalg.eigs(self.adj_mat(R=self.R), k=config.n_eigs)
        self.eigenvalues = self.eigenvalues[:config.n_eigs]
        self.eigenvectors = self.eigenvectors[:, :config.n_eigs]

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

    def build_generator(self):
        """initializing the generator"""

        with tf.variable_scope("generator"):
            self.generator = generator.Generator(n_node=self.n_node, n_layer=config.n_layers)

    def build_discriminator(self):
        """initializing the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(data=data, n_node=self.n_node, n_layer=config.n_layers)

    def train(self):

        print("start training...")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)

            # D-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            labels = []
            losess = []
            for d_epoch in range(config.n_epochs_dis):
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % config.dis_interval == 0:
                    eigen_vectors, eigen_values, node_1, node_2, labels = self.prepare_data_for_d()
                _, loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss],
                              feed_dict={self.discriminator.eigen_vectors: eigen_vectors,
                                         self.discriminator.eigen_values: eigen_values,
                                         self.discriminator.node_id: np.array(node_1),
                                         self.discriminator.node_neighbor_id: np.array(node_2),
                                         self.discriminator.label: np.array(labels)})
                losess.append(loss)
            print("d_loss %f" % np.mean(np.asarray(losess)))

            # G-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            reward = []
            losess = []
            for g_epoch in range(config.n_epochs_gen):
                if g_epoch % config.gen_interval == 0:
                    eigen_vectors, eigen_values, node_1, node_2, reward = self.prepare_data_for_g()

                _, loss = self.sess.run([self.generator.g_updates, self.generator.loss],
                                feed_dict={self.generator.eigen_vectors: eigen_vectors,
                                           self.generator.eigen_values: eigen_values,
                                           self.generator.node_id: np.array(node_1),
                                           self.generator.node_neighbor_id: np.array(node_2),
                                           self.generator.reward: np.array(reward)})
                losess.append(loss)
            print("g_loss %f" % np.mean(np.asarray(losess)))

            ret = test.test(sess=self.sess, model=self.generator, users_to_test=list(data.test_set.keys()))
            print('gen_recall_20 %f gen_recall_40 %f gen_recall_60 %f gen_recall_80 %f gen_recall_100 %f'
                  % (ret[0], ret[1], ret[2], ret[3], ret[4]))
            print('gen_map_20 %f gen_map_40 %f gen_map_60 %f gen_map_80 %f gen_map_100 %f'
                  % (ret[5], ret[6], ret[7], ret[8], ret[9]))
            ret = test.test(sess=self.sess, model=self.discriminator, users_to_test=list(data.test_set.keys()))
            print('dis_recall_20 %f dis_recall_40 %f dis_recall_60 %f dis_recall_80 %f dis_recall_100 %f'
                  % (ret[0], ret[1], ret[2], ret[3], ret[4]))
            print('dis_map_20 %f dis_map_40 %f dis_map_60 %f dis_map_80 %f dis_map_100 %f'
                  % (ret[5], ret[6], ret[7], ret[8], ret[9]))
        print("training completes")

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        users = random.sample(range(self.n_users), config.missing_edge)

        pos_items = []
        for u in users:
            p_items = set(np.nonzero(self.R[u, :])[0].tolist())
            p_item = random.sample(p_items, 1)[0]
            pos_items.append(p_item)

        node_2 = [self.n_users + p for p in pos_items]

        all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.eigen_vectors: self.eigenvectors,
                                                                       self.generator.eigen_values: self.eigenvalues})
        negative_items = []
        for u in users:
            neg_items = list(set(range(self.n_items)) - set(np.nonzero(self.R[u, :])[0].tolist()))

            relevance_probability = all_score[u, neg_items]
            relevance_probability = utils.softmax(relevance_probability)
            neg_item = np.random.choice(neg_items, size=1, p=relevance_probability)[0]  # select next node
            negative_items.append(neg_item)

        node_2 += [self.n_users + p for p in negative_items]
        node_1 = users*2
        labels = [1.0]*config.missing_edge + [0.0] * config.missing_edge

        return self.eigenvectors, self.eigenvalues, node_1, node_2, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        users = random.sample(range(self.n_users), config.missing_edge)

        #eigenvalues, eigenvectors = linalg.eigs(adj_missing, k=config.n_eigs)

        all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.eigen_vectors: self.eigenvectors,
                                                                       self.generator.eigen_values:   self.eigenvalues})

        negative_items = []
        for u in users:
            pos_items = set(np.nonzero(self.R[u, :])[0].tolist())
            neg_items = list(set(range(self.n_items)) - pos_items)
            relevance_probability = all_score[u, neg_items]
            relevance_probability = utils.softmax(relevance_probability)
            neg_item = np.random.choice(neg_items, size=1, p=relevance_probability)[0]  # select next node
            negative_items.append(data.n_users + neg_item)

        node_1 = users*2
        node_2 = negative_items*2
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.eigen_vectors: self.eigenvectors,
                                          self.discriminator.eigen_values: self.eigenvalues,
                                          self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return self.eigenvectors, self.eigenvalues, node_1[:config.missing_edge], node_2[:config.missing_edge], reward[:config.missing_edge]


    def adj_mat(self, R, self_connection=True):
        A = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items], dtype=np.float32)
        A[:self.n_users, self.n_users:] = R
        A[self.n_users:, :self.n_users] = R.T
        if self_connection:
            return np.identity(self.n_users + self.n_items, dtype=np.float32) + A
        return A


if __name__ == "__main__":
    spectral_gan = SpectralGAN(n_users=data.n_users, n_items=data.n_items, R=data.R)
    spectral_gan.train()
