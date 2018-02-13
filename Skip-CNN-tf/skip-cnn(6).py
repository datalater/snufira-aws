# -*- coding: utf-8 -*-

data_path = '../data/books_text_full/test/'

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

# 텍스트 파일의 내용을 변수 text로 리턴하는 함수
def load_doc(filename):
    # read only로 파일을 엽니다.
    file = open(filename, 'r', errors='replace')
    # 모든 텍스트를 읽습니다.
    text = file.read()
    # 파일을 닫습니다.
    file.close()
    return text

def clean_doc(doc):
    # white space 기준으로 tokenize 합니다.
    tokens = doc.split()
    # 각 token에서 모든 구두점을 삭제합니다.
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # 각 token에서 alaphabet으로만 이루어지지 않은 모든 단어를 삭제합니다.
    tokens = [word for word in tokens if word.isalpha()]
    # 각 token에서 stopwrods를 삭제합니다.
    stop_words = set(stopwords.words('english'))
    tokens = [w.lower() for w in tokens if not w in stop_words]
    # 각 token에서 1글자 이하인 모든 단어를 삭제합니다.
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# 텍스트 파일을 불러와서 vocab에 추가하는 함수
def add_doc_to_vocab(filename, vocab):
    # 텍스트 파일을 불러옵니다.
    doc = load_doc(filename)
    # 텍스트 파일을 clean toekn으로 리턴합니다.
    tokens = clean_doc(doc)
    # clean token을 vocab에 추가합니다.
    vocab.update(tokens)

# 폴더에 있는 모든 문서를 vocab에 추가하는 함수
def process_docs(directory, vocab, is_train):
    # 폴더에 있는 모든 파일을 순회합니다.
    for filename in listdir(directory):
        # 인덱스가 새겨진 파일 이름과 is_train 인자를 기준으로 test set으로 분류할 모든 파일을 건너뜁니다.
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # 폴더에 있는 파일의 절대 경로를 구합니다.
        path = directory + '/' + filename
        # 텍스트 파일을 불러와서 vocab에 추가하는 함수를 실행합니다.
        add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
    # 각 문장을 하나의 텍스트 일부로 바꿉니다.
    data = '\n'.join(lines)
    # 파일을 쓰기 모드로 엽니다.
    file = open(filename, 'w')
    # 변환한 텍스트를 파일에 씁니다.
    file.write(data)
    # 파일을 닫습니다.
    file.close()

# vocab을 Counter() 객체로 할당합니다.
vocab = Counter()
# 폴더를 지정하고 폴더 내 모든 문서를 vocab에 추가합니다.
process_docs(data_path, vocab, True)
# vocab의 크기를 출력합니다.
# print(len(vocab))
# vocab에서 가장 많이 등장한 50개 단어를 출력합니다.
# print(vocab.most_common(50))

# token을 min_occurence 기준으로 유지합니다.
min_occurence = 1
tokens = [k for k,c in vocab.items() if c >= min_occurence]
print(len(tokens))
# token을 vocab 파일로 저장합니다.
save_list(tokens, 'corpusToLines_vocab.txt')
# print("\n# 단어 {}개의 [corpusToLines_vocab.txt]로 저장했습니다.".format(len(tokens)))

# 보카를 불러옵니다.
vocab_filename = 'corpusToLines_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# print("# 단어 {}개의 [{}]을 [vocab]으로 불러왔습니다.".format(len(vocab), vocab_filename))


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
print("# 문장 {}개가 [vocab_lines.txt]로 저장되었습니다.".format(len(vocab_lines)))
filename = 'vocab_lines.txt'
vocab_lines = load_doc(filename)
vocab_lines = [i for i in vocab_lines.splitlines()]
vocab_vocab = set()
for i in vocab_lines:
    vocab_vocab.update(i)
print("# unique words in [vocab_lines.txt]: [{}]".format(len(vocab_vocab)))

##############################################################################

filename = "vocab_lines.txt"

file = open(filename, 'r', errors='replace')
text = file.read()
file.close()

vocab_lines = [i for i in text.splitlines()]

list_lines = []
for i in vocab_lines:
    i = i.split()
    list_lines.append(i)

print(list_lines[0])

sentences = list_lines
print("Total training sentences:{}".format(len(sentences)))

from gensim.models import Word2Vec

wv_sz = 100
# word2vec 모델을 훈련시킵니다.
model = Word2Vec(sentences, size=wv_sz, window=5, workers=8, min_count=1)
# 모델의 vocabulary size를 요약합니다.
words = list(model.wv.vocab)
print("Vocabulary size: %d" % len(words))
print("Wordvector size: %d" % (wv_sz))
print("Embedding size: {}x{}".format(len(words), wv_sz))

# 모델을 ASCII 포맷으로 저장합니다.
filename = 'fantasy_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
print("\n# word2vec 파일 [{}]이 저장되었습니다.".format(filename))

##############################################################################

max_length = max([len(s.split()) for s in vocab_lines])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
encoded_lines = np.array(list(vocab_processor.fit_transform(vocab_lines)))
# print("-"*80,"# [vocab_lines]가 [encoded_lines]로 인코딩 및 패딩 되었습니다. (max_length:{})".format(max_length), "-"*80, sep='\n')
# print("BEFORE: \n{}".format(vocab_lines[0]))
# print("\nAFTER: \n{}".format(encoded_lines[0]))

x_data = np.array(list(encoded_lines))
# print("\n", "-"*80,"# 최종 [x_data] (max_length: {})".format(max_length), "-"*80, sep='\n')
# print("EXAMPLE: \n{}".format(x_data[0]))

vocab_dict = vocab_processor.vocabulary_._mapping
vocab_size = len(vocab_dict.keys())
# print("\n", "-"*80,"# 최종 [vocab_dict] (vocab_size: {})".format(vocab_size), "-"*80, sep='\n')
# print("EXAMPLE: \n[{}] is mapped to [{}].".format(vocab_lines[0].split()[0], vocab_dict[vocab_lines[0].split()[0]]))

##############################################################################

def load_word2vec(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    lines = file.readlines()[1:]
    for line in lines:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded {}!'.format(filename))
    file.close()
    return vocab,embd

filename = './fantasy_embedding_word2vec.txt'
vocab,embd = load_word2vec(filename)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
print(len(embedding))

sequence_length = max_length
batch_size = 32
num_sentence = batch_size

def cossim(a, b):
    dot=tf.cast(tf.tensordot(a, b, axes=1), tf.float32)

    norm1=tf.sqrt(tf.cast(tf.tensordot(a, a, axes=1), tf.float32))
    norm2=tf.sqrt(tf.cast(tf.tensordot(b, b, axes=1), tf.float32))

    mycossi=tf.div(dot, tf.multiply(norm1, norm2))

    return mycossi

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

loss = tf.zeros([])
for si in range(num_sentence - 3):
    for ssi in range(si, num_sentence - 2):
        if not si == ssi:

            cossi1 = cossim(cnn_output[si], cnn_output[ssi])
            cossi2 = cossim(cnn_output[si+1], cnn_output[ssi+2])
            cossi3 = cossim(cnn_output[si+2], cnn_output[ssi+2])

            cossi= tf.abs(tf.subtract(tf.div(tf.add(cossi1, cossi3),2), cossi2))

            loss= tf.add(loss,cossi)

loss = tf.reshape(loss, [])

tf.summary.scalar("loss", loss)
# tf.summary.scalar("W", W)
# tf.scalar_summary('b', b)

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
