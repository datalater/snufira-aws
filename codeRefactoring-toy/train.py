# -*- coding: utf-8 -*-

import vocab as vc
import data_helpers as dh
import custom_loss as cl

from collections import Counter
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

data_path = '../data/books_text_full/test/'


# Make vocabulary
# ===========================================================================

print("# Make vocabulary")
print("#", "="*70)

vocab = Counter()
vc.process_docs(data_path, vocab)

min_occurence = 1
tokens = [k for k,c in vocab.items() if c >= min_occurence]

vc.save_list(tokens, 'corpusToLines_vocab.txt')
print("# {} words are stored to [corpusToLines_vocab.txt].".format(len(tokens)))

vocab_filename = 'corpusToLines_vocab.txt'
vocab = vc.load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
print("# [{}] is loaded as [vocab].".format(vocab_filename))


# Data preparation
# ===========================================================================

print("\n# Data preparation")
print("#", "="*70)

sentences = dh.process_directory(data_path)
dh.save_list(sentences, 'total_lines.txt')
print("# 문장 {}개가 [total_lines.txt]로 저장되었습니다.".format(len(sentences)))
filename = 'total_lines.txt'

filename = "total_lines.txt"
clean_lines = dh.doc_to_clean_lines(filename)
dh.save_list(clean_lines, 'clean_lines.txt')
print("# 문장 {}개가 [clean_lines.txt]로 저장되었습니다.".format(len(clean_lines)))

filename = "clean_lines.txt"
vocab_lines = dh.doc_to_vocab_lines(filename, vocab)
dh.save_list(vocab_lines, 'vocab_lines.txt')
print("# 문장 {}개가 [vocab_lines.txt]로 저장되었습니다.".format(len(vocab_lines)))

max_length = max([len(s.split()) for s in vocab_lines])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
encoded_lines = np.array(list(vocab_processor.fit_transform(vocab_lines)))
# print("# [vocab_lines]가 [encoded_lines]로 인코딩 및 패딩 되었습니다. (max_length:{})".format(max_length))
# print("BEFORE: \n{}".format(vocab_lines[0]))
# print("\nAFTER: \n{}".format(encoded_lines[0]))

x_data = np.array(list(encoded_lines))
print("# [vocab_lines]가 [x_data]로 인코딩 및 패딩 되었습니다. (max_length:{})".format(max_length))
# print("# 최종 [x_data] (max_length: {})".format(max_length))
# print("EXAMPLE: \n{}".format(x_data[0]))

# vocab_dict = vocab_processor.vocabulary_._mapping
# vocab_size = len(vocab_dict.keys())
# print("\n", "-"*80,"# 최종 [vocab_dict] (vocab_size: {})".format(vocab_size), "-"*80, sep='\n')
# print("EXAMPLE: \n[{}] is mapped to [{}].".format(vocab_lines[0].split()[0], vocab_dict[vocab_lines[0].split()[0]]))


# Word2Vec
# ===========================================================================

print("\n# Word2Vec")
print("#", "="*70)

filename = "vocab_lines.txt"
text = dh.load_doc(filename)

vocab_lines = [i for i in text.splitlines()]

list_lines = []
for i in vocab_lines:
    i = i.split()
    list_lines.append(i)

sentences = list_lines
print("# Total training sentences:{}".format(len(sentences)))

wv_sz = 100
# word2vec 모델을 훈련시킵니다.
model = Word2Vec(sentences, size=wv_sz, window=5, workers=8, min_count=1)
# 모델의 vocabulary size를 요약합니다.
words = list(model.wv.vocab)
print("# Vocabulary size: %d" % len(words))
print("# Wordvector size: %d" % (wv_sz))
print("# Embedding size: {}x{}".format(len(words), wv_sz))

# 모델을 ASCII 포맷으로 저장합니다.
filename = 'fantasy_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
print("# word2vec 파일 [{}]이 저장되었습니다.".format(filename))


# Load Embedding
# ===========================================================================

print("\n# Load embedding")
print("#", "="*70)

filename = './fantasy_embedding_word2vec.txt'
vocab,embd = dh.load_word2vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
print("# embedding vocabulary size: {}".format(len(embedding)))


# Training
# ===========================================================================

print("\n# Training")
print("#", "="*70)

sequence_length = max_length
batch_size = 32
num_sentence = batch_size

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
tf_embedding = tf.constant(embedding, dtype=tf.float32)

embedded_chars = tf.nn.embedding_lookup(tf_embedding, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

filter_sizes = [3, 4, 5]
num_filters = 128

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, wv_sz, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
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

            cossi1 = cl.cossim(cnn_output[si], cnn_output[ssi])
            cossi2 = cl.cossim(cnn_output[si+1], cnn_output[ssi+2])
            cossi3 = cl.cossim(cnn_output[si+2], cnn_output[ssi+2])

            cossi = tf.subtract(tf.div(tf.add(cossi1, cossi3),2), cossi2)

            loss = tf.add(loss,cossi)

loss = tf.reshape(loss, [])
tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
merge_op = tf.summary.merge_all()

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

        if i % batch_size == 0:
            summary = sess.run(merge_op, feed_dict={input_x: x_batch})
            summary_writer.add_summary(summary, k)
        k += 1

    print("Epoch: %04d" % (epoch + 1))
    print("Avg. cost: {}".format(total_loss / total_batch))

print("최적화 완료!")

# tensorboard --logdir="./logs" --port=9000
