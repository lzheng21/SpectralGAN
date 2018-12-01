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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SpectralGAN(object):
    def __init__(self, n_users, n_items, R):
        print("reading graphs...")
        self.n_users, self.n_items = n_users, n_items
        self.n_node = n_users + n_items
        self.R = R

        # construct graph
        self.R = R

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
            self.generator = generator.Generator(n_node=self.n_node, n_layer=1)

    def build_discriminator(self):
        """initializing the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(n_node=self.n_node, n_layer=1)

    def train(self):

        print("start training...")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)

            # D-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            labels = []
            for d_epoch in range(config.n_epochs_dis):
                # generate new nodes for the discriminator for every dis_interval iterations
                if d_epoch % config.dis_interval == 0:
                    adj_missing, node_1, node_2, labels = self.prepare_data_for_d()
                self.sess.run(self.discriminator.d_updates,
                              feed_dict={self.discriminator.adj_miss: np.array(adj_missing),
                                         self.discriminator.node_id: np.array(node_1),
                                         self.discriminator.node_neighbor_id: np.array(node_2),
                                         self.discriminator.label: np.array(labels)})

            # G-steps
            adj_missing = []
            node_1 = []
            node_2 = []
            reward = []
            for g_epoch in range(config.n_epochs_gen):
                if g_epoch % config.gen_interval == 0:
                    adj_missing, node_1, node_2, reward = self.prepare_data_for_g()

                self.sess.run(self.generator.g_updates,
                                feed_dict={self.generator.adj_miss: np.array(adj_missing),
                                           self.generator.node_id: np.array(node_1),
                                           self.generator.node_neighbor_id: np.array(node_2),
                                           self.generator.reward: np.array(reward)})


            ret = test.test(sess=self.sess, model=self.generator, users_to_test=data.test_set.keys())
            print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
                  % (ret[0], ret[1], ret[2], ret[3], ret[4]))
            print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
                  % (ret[5], ret[6], ret[7], ret[8], ret[9]))
        print("training completes")

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        adj_mats = []
        node_1 = []
        node_2 = []
        labels = []
        users = random.sample(range(self.n_users), int(config.batch_size_dis/2))
        for u in users:
            pos_items = set(np.nonzero(self.R[u, :])[0].tolist())
            pos_item = random.sample(pos_items, 1)[0]
            R_missing = self.R
            R_missing[u, pos_item] = 0
            adj_missing = self.adj_mat(R=R_missing)

            adj_mats.append(adj_missing)
            node_1.append(u)
            node_2.append(self.n_users + pos_item)
            labels.append(1.0)

        adj_mats = adj_mats * 2
        all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.adj_miss: np.array(adj_mats)})
        for u in users:
            pos_items = set(np.nonzero(self.R[u, :])[0].tolist())
            neg_items = list(set(range(self.n_items)) - pos_items)

            relevance_probability = all_score[u, neg_items]
            relevance_probability = utils.softmax(relevance_probability)
            neg_item = np.random.choice(neg_items, size=1, p=relevance_probability)[0]  # select next node

            node_1.append(u)
            node_2.append(self.n_users + neg_item)
            labels.append(0.0)


        return adj_mats, node_1, node_2, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""

        adj_mats = []
        node_1 = []
        node_2 = []
        for u in random.sample(range(self.n_users), config.batch_size_gen):
            pos_items = set(np.nonzero(self.R[u,:])[0].tolist())
            neg_items = set(range(self.n_items)) - pos_items
            pos_item = random.sample(pos_items, 1)[0]

            R_missing = self.R
            R_missing[u, pos_item] = 0
            adj_missing = self.adj_mat(R=R_missing)
            adj_mats.append(adj_missing)
            node_1.append(u)

        all_score = self.sess.run(self.generator.all_score, feed_dict={self.generator.adj_miss: np.array(adj_mats)})
        for u in node_1:
            pos_items = set(np.nonzero(self.R[u, :])[0].tolist())
            neg_items = list(set(range(self.n_items)) - pos_items)
            relevance_probability = all_score[u, neg_items]
            relevance_probability = utils.softmax(relevance_probability)
            neg_item = np.random.choice(neg_items, size=1, p=relevance_probability)[0]  # select next node
            node_2.append(self.n_users+neg_item)


        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.adj_miss: np.array(adj_mats),
                                          self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return adj_mats, node_1, node_2, reward


    def adj_mat(self, R, self_connection=True):
        A = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items], dtype=np.float32)
        A[:self.n_users, self.n_users:] = R
        A[self.n_users:, :self.n_users] = R.T
        if self_connection:
            return np.identity(self.n_users + self.n_items, dtype=np.float32) + A
        return A


if __name__ == "__main__":
    data = load_data.Data(train_file=config.train_filename, test_file=config.test_filename)
    spectral_gan = SpectralGAN(n_users=data.n_users, n_items=data.n_items, R=data.R)
    spectral_gan.train()