import tensorflow as tf
import numpy as np

class rl_model(object):

    def __init__(self, learning_rate, discount_factor, seed):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.session = tf.Session()
        self.set_model()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def _add_layer(self, in_dim, out_dim, input):
        w = tf.Variable(initial_value=tf.random_normal(shape=(in_dim, out_dim), dtype=tf.float32))
        b = tf.Variable(initial_value=tf.zeros(shape=(1,out_dim), dtype=tf.float32))

        return tf.matmul(input, w) + b
     
    def save(self, name):
        return self.saver.save(self.session, "./tmp/model_%3.f.ckpt" % (name))
    
    def load(self, name):
        self.saver.restore(self.session, "./tmp/model_%3.f.ckpt" % (name))

    def set_model(self):
        self._x = tf.placeholder(tf.float32, shape=(None, 4), name="X")
        self._y = tf.placeholder(tf.float32, shape=(None, 2), name="Y")

#         z1 = self._add_layer(4, 16, self._x)
#         a1 = tf.nn.relu(z1)
#         z2 = self._add_layer(16, 16, a1)
#         a2 = tf.nn.relu(z2)
#         self.y_ = self._add_layer(16, 2, a2)

        z1 = self._add_layer(4, 8, self._x)
        a1 = tf.nn.relu(z1)
        z2 = self._add_layer(8, 8, a1)
        a2 = tf.nn.relu(z2)
        self.y_ = self._add_layer(8, 2, a2)

#         self.y_ = self._add_layer(4, 2, self._x)

        self.loss = tf.losses.mean_squared_error(self._y, self.y_)
        self.step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def decrease_learning_rate(self):
        self.lr *= 0.3

    def predict(self, states):
        return self.session.run(self.y_, feed_dict={self._x:states})

    def _preprocess(self, data):
        y_next = np.max(self.predict(data[3]), axis=1)
        y_target = self.predict(data[0])
        for i in range(len(data[0])):
            y_target[i, data[1][i]] = data[2][i] + self.discount_factor * y_next[i]
        return data[0], y_target

    def train(self, data):
        x, y = self._preprocess(data)
        loss, _ = self.session.run([self.loss, self.step], 
                                   feed_dict={self._x:x, self._y:y})
        return loss