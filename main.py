import tensorflow as tf
import os
from data_helper import read_vocab_to_dict, build_word2vec, dataset_iterator, load_verb_count
from model import Model

# read word and verb dict
print("load dictionary...")
word_dict = read_vocab_to_dict("data/word_vocab.txt")
verb_dict = read_vocab_to_dict("data/verb_vocab.txt")
verb_vocab_count = load_verb_count("data/verb_count.txt")

flags = tf.flags
flags.DEFINE_integer("neg_sample", 10, "number of negative samples")
flags.DEFINE_integer("word_dim", 300, "word embedding dimension")
flags.DEFINE_integer("num_units", 100, "number of units for rnn cell and hidden layer of ffn")
flags.DEFINE_integer("output_units", 100, "number of units for output part")
flags.DEFINE_bool("use_pretrained", True, "use pretrained word2vec")
flags.DEFINE_bool("tune_emb", False, "tune pretrained embeddings while training")
flags.DEFINE_string("pretrained_emb", "data/word2vec.npz", "pretrained embedding path")
flags.DEFINE_integer("vocab_size", len(word_dict), "word vocab size")
flags.DEFINE_integer("verb_size", len(verb_dict), "verb vocab size")
flags.DEFINE_float("lr", 0.001, "learning_rate")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_string("dataset", "data/dataset_10k.txt", "dataset")
flags.DEFINE_integer("epochs", 3, "epochs")
config = flags.FLAGS

# build pretrained word2vec embeddings if possible
if not os.path.exists(config.pretrained_emb):
    build_word2vec(None, word_dict, config.word_dim)

# build model
print("build model...")
model = Model(config, verb_vocab_count)

out_file = open("data/log.txt", mode="w")

# training the model
print("start training...")
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    for epoch in range(config.epochs):
        for i, data in enumerate(dataset_iterator(config.dataset, word_dict, verb_dict, config.batch_size)):
            feed_dict = model.get_feed_dict(data, is_train=True, lr=config.lr)
            _, losses, neg_ids = sess.run([model.train_op, model.loss, model.neg_ids], feed_dict=feed_dict)
            print("epoch: {}, batch_id: {}, neg_ids: {}".format(epoch, i, neg_ids))
            out_file.write("epoch: {}, batch_id: {}, neg_ids: {}\n".format(epoch, i, neg_ids))
            if i % 10 == 0:
                print(losses)
    saver.save(sess, "ckpt/mode", global_step=config.epochs)

out_file.close()
