
# Vocabulary
# ========================================================

def load_doc(filename):
    file = open(filename, 'r', errors='ignore')
    text = file.read()
    file.close()
    return text

filename = '/bookcorpus/total_vocab.txt'
vocab = load_doc(filename)
vocab = vocab.split()
vocab = set(vocab)

print("# {} words is loaded as [{}] with [vocab].".format(len(vocab), vocab_filename))

# Corpus to lines
# ========================================================

def doc_to_clean_lines(filename):
    total_lines = load_doc(filename)
    clean_lines = [i.lower() for i in total_lines.splitlines() if len(i) > 5 if "." in i]

    return clean_lines

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

filename = "/bookcorpus/total_lines.txt"
clean_lines = doc_to_clean_lines(filename)
save_list(clean_lines, 'clean_lines.txt')
print("# {} sentences are stored as [clean_lines.txt].".format(len(clean_lines)))
filename = 'clean_lines.txt'
clean_lines = load_doc(filename)
clean_lines = [i for i in clean_lines.splitlines()]
clean_vocab = set()
for i in clean_lines:
    clean_vocab.update(i)
print("# unique words in [clean_lines.txt]: [{}]".format(len(clean_vocab)))

def doc_to_vocab_lines(filename):
    clean_lines = load_doc(filename)
    vocab_lines = []
    for i in clean_lines.splitlines():
        words = i.split()
        words = [word for word in words if word in vocab]
        words = [word for word in words if len(words) >= 5]
        vocab_line = " ".join(words)
        if len(vocab_line):
            vocab_line += "."
            vocab_line = [vocab_line]
            vocab_lines += vocab_line

    return vocab_lines

filename = "clean_lines.txt"
vocab_lines = doc_to_vocab_lines(filename)
save_list(vocab_lines, 'vocab_lines.txt')
print("# {} sentences are stored as [vocab_lines.txt].".format(len(vocab_lines)))
filename = 'vocab_lines.txt'
vocab_lines = load_doc(filename)
vocab_lines = [i for i in vocab_lines.splitlines()]
vocab_vocab = set()
for i in vocab_lines:
    vocab_vocab.update(i)
print("# unique words in [vocab_lines.txt]: [{}]".format(len(vocab_vocab)))

import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

max_length = max([len(s.split()) for s in vocab_lines])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
encoded_lines = np.array(list(vocab_processor.fit_transform(vocab_lines)))
print("-"*80,"# [vocab_lines]is transformed into [encoded_lines]. (max_length:{})".format(max_length), "-"*80, sep='\n')
print("BEFORE: \n{}".format(vocab_lines[0]))
print("\nAFTER: \n{}".format(encoded_lines[0]))

x_data = np.array(list(encoded_lines))
print("\n", "-"*80,"# final [x_data] (max_length: {})".format(max_length), "-"*80, sep='\n')
print("EXAMPLE: \n{}".format(x_data[0]))

vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict.keys())
print("\n", "-"*80,"# final [vocab_dict] (vocab_size: {})".format(vocab_size), "-"*80, sep='\n')
print("EXAMPLE: \n[{}] is mapped to [{}].".format(vocab_lines[0].split()[0], vocab_dict[vocab_lines[0].split()[0]]))



# Load word embedding
# ========================================================

def load_word2vec(filename):
    vocab = []
    embd = []
    file = open(filename,'r', errors='ignore')
    lines = file.readlines()[1:]
    for line in lines:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('# [{}] is successfully loaded!'.format(filename))
    file.close()
    return vocab,embd

filename = '/bookcorpus/fantasy_embedding_word2vec.txt'
vocab,embd = load_word2vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
print("# embedding vocabulary size: {}".format(len(embedding)))

sequence_length = max_length

batch_size = 32
num_sentence = batch_size

def cossim(a, b):
    dot=tf.cast(tf.tensordot(a, b, axes=1), tf.float32)

    norm1=tf.sqrt(tf.cast(tf.tensordot(a, a, axes=1), tf.float32))
    norm2=tf.sqrt(tf.cast(tf.tensordot(b, b, axes=1), tf.float32))

    mycossi=tf.div(dot, tf.multiply(norm1, norm2))

    return mycossi

global_step = tf.Variable(0, trainable=False, name='global_step')

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
tf_embedding = tf.constant(embedding, dtype=tf.float32)

embedded_chars = tf.nn.embedding_lookup(tf_embedding, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

filter_sizes = [3, 4, 5]
num_filters = 128
wv_sz = 100

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, wv_sz, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat( pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

cnn_output = h_pool_flat

loss = tf.zeros(1)
for si in range(num_sentence - 3):
    for ssi in range(si, num_sentence-2):
        if not si == ssi:

            cossi1 = cossim(cnn_output[si], cnn_output[ssi])
            cossi2 = cossim(cnn_output[si+1], cnn_output[ssi+2])
            cossi3 = cossim(cnn_output[si+2], cnn_output[ssi+2])

            cossi = tf.abs(tf.subtract(tf.div(tf.add(cossi1, cossi3),2), cossi2))

            loss = tf.add(loss,cossi)

loss = tf.reshape(loss, [])

tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step)
merge_op = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.Session()
summary_writer = tf.summary.FileWriter("./logs", sess.graph)
sess.run(tf.global_variables_initializer())

batch_size = 32
total_batch = int(len(x_data) / batch_size)

for epoch in range(5):
    total_loss = 0
    k = 0

    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i+batch_size]

        _, loss_val = sess.run([optimizer,loss],
                                feed_dict={input_x: x_batch})
        total_loss += loss_val

        summary = sess.run(merge_op, feed_dict={input_x: x_batch})
        summary_writer.add_summary(summary, global_step=sess.run(global_step))

    print("Epoch: %04d" % (epoch + 1))
    print("Avg. cost: {}".format(total_loss / total_batch))

print("Optimization finished!")
save_path = saver.save(sess, "./logs/model3.ckpt")
print("Model is successfully saved!")
