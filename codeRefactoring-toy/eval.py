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

total_lines = dh.process_directory(data_path)
dh.save_list(total_lines, 'total_lines.txt')
print("# 문장 {}개가 [total_lines.txt]로 저장되었습니다.".format(len(total_lines)))

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

x_data = np.array(list(encoded_lines))
print("# [vocab_lines]가 [x_data]로 인코딩 및 패딩 되었습니다. (max_length:{})".format(max_length))


# Parameters
# ===========================================================================

print("\n# Parameters")
print("#", "="*70)

# Eval Parameters
x_test = x_data

# Misc Parameters

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Evaluation
# ===========================================================================

print("\n# Evaluation")
print("#", "="*70)

checkpoint_file = checkpoint_file = tf.train.latest_checkpoint("./logs")

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = dh.batch_iter(list(x_test), 32, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

#
