import os
import codecs
import math
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
import gensim

PAD = "<pad>"
UNK = "<unk>"

# cf. https://courses.washington.edu/hypertxt/csar-v02/penntable.html
# cf. http://wacky.sslmit.unibo.it/lib/exe/fetch.php?media=tagsets:ukwac_tagset.txt
verb_pos = ["VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VV", "VVN", "VVG", "VVD", "VVZ", "VVP", "VH", "VHP", "VHZ", "VHD",
            "VHG", "VHN"]

separate = "\t"
neg_separate = " , "

word2vec_path = os.path.join(os.path.expanduser("~"), "data", "embeddings", "GoogleNews-vectors-negative300.bin")


def build_word2vec(filename, word_dict, dim):
    scale = np.sqrt(3.0 / dim)
    embeddings = np.random.uniform(-scale, scale, [len(word_dict), dim])
    if filename is None:
        filename = word2vec_path
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    for word in w2v_model.wv.vocab:
        if word in word_dict:
            idx = word_dict[word]
            embeddings[idx] = w2v_model.wv[word]
        elif word.lower() in word_dict:
            idx = word_dict[word.lower()]
            embeddings[idx] = w2v_model.wv[word]
    np.savez_compressed("data/word2vec.npz", embeddings=embeddings)


def ukwac_corpus_iterator(ukwac_path=None, lowercase=True):
    if ukwac_path is None:
        ukwac_path = os.path.join(os.path.expanduser("~"), "data", "ukwac", "ukwac_pos", "pos_text")
    ukwac_file = os.path.join(ukwac_path, "UKWAC-{}.xml")
    if not os.path.exists(ukwac_path):
        raise IOError("Unable to find the corpus directory: %s" % ukwac_path)
    file_indices = list(range(1, 26))  # 26
    for file_index in file_indices:
        words, poss, lemmas = [], [], []
        total = int(3.34e7) if file_index == 25 else int(1e8)
        with codecs.open(ukwac_file.format(file_index), mode="r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, total=total, desc="Read UKWAC-{}.xml".format(file_index)):
                line = line.lstrip().rstrip()
                if line.startswith("<text id") or line.startswith("<s>") or line.startswith("</text>"):
                    continue
                if line.startswith("</s>"):  # read one sentence
                    if len(words) > 80:  # if sentence length is greater than 80, then ignore this sentence
                        words, poss, lemmas = [], [], []
                        continue
                    yield words, poss, lemmas
                    words, poss, lemmas = [], [], []
                    continue
                if len(line.split("\t")) != 3:
                    continue
                word, pos, lemma = line.split("\t")
                words.append(word.strip().lower() if lowercase else word.strip())
                poss.append(pos.strip())
                lemmas.append(lemma.strip().lower() if lowercase else lemma.strip())


def build_dataset_and_vocab(save_path, word_threshold=90):
    word_counter, verb_counter = Counter(), Counter()
    save_file = codecs.open(os.path.join(save_path, "dataset.txt"), mode="w", encoding="utf-8")
    for words, poss, _ in ukwac_corpus_iterator():
        for idx, word in enumerate(words):
            # count all words
            word_counter[word] += 1
            # count verbs
            if poss[idx] in verb_pos:
                verb_counter[word] += 1
            # build dataset
            if idx != 0 and idx != len(words) - 1 and poss[idx] in verb_pos:
                left_context = " ".join(words[:idx])
                right_context = " ".join(words[idx + 1:])
                save_file.write(separate.join([left_context, word, right_context]) + "\n")
                save_file.flush()
    save_file.close()

    # build word vocabulary
    word_vocab = [PAD] + [word for word, count in word_counter.most_common() if count >= word_threshold] + [UNK]

    # create negative sampling table
    negative_table = build_negative_sampling_table(verb_counter, table_size=int(1e8), threshold=word_threshold)

    # save to file
    with codecs.open(os.path.join(save_path, "word_vocab.txt"), mode="w", encoding="utf-8") as f:
        f.write("\n".join(word_vocab))
    with codecs.open(os.path.join(save_path, "verb_count.txt"), mode="w", encoding="utf-8") as f:
        f.write("\n".join(["{}\t{}".format(word, count) for word, count in verb_counter.most_common()]))
    with codecs.open(os.path.join(save_path, "neg_table.txt"), mode="w", encoding="utf-8") as f:
        f.write("\n".join(negative_table))
    return negative_table


def build_negative_sampling_table(verb_freq_dict, table_size=int(1e8), threshold=90):
    if isinstance(verb_freq_dict, str):
        tmp_dict = Counter()
        with codecs.open(verb_freq_dict, mode="r", encoding="utf-8") as f:
            for line in f:
                verb, count = line.strip().split("\t")
                tmp_dict[verb] += int(count)
        verb_freq_dict = tmp_dict.copy()
    verb_freq_dict = dict([(verb, math.pow(count, 0.75)) for verb, count in verb_freq_dict.most_common()
                           if count >= threshold])
    sum_freq = sum([freq for _, freq in verb_freq_dict.items()])
    verb_freq_dict = dict([(verb, freq / sum_freq) for verb, freq in verb_freq_dict.items()])
    negative_table = []
    for verb, freq in verb_freq_dict.items():
        negative_table += [verb] * int(freq * table_size)
    random.shuffle(negative_table)
    return negative_table


def negative_sampling(filename, file_out, negative_table, num_neg=10):
    np.random.seed(123456)
    if filename is None or not os.path.exists(filename):
        raise IOError("Unable to find the dataset file: %s" % filename)
    neg_size = len(negative_table)
    f_out = codecs.open(file_out, mode="w", encoding="utf-8")
    with codecs.open(filename, mode="r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Read dataset"):
            l_context, verb, r_context = line.strip().split("\t")
            # negative sampling
            neg_verbs = []
            for _ in range(num_neg):
                idx = np.random.randint(neg_size)
                while negative_table[idx] == verb or negative_table[idx] in neg_verbs:
                    idx = np.random.randint(neg_size)
                neg_verbs.append(negative_table[idx])
            f_out.write(separate.join([l_context, verb, r_context, neg_separate.join(neg_verbs)]) + "\n")
            f_out.flush()
    f_out.close()


def negative_sample(verb, negative_table, neg_size, num_neg=10):
    neg_verbs = []
    for _ in range(num_neg):
        idx = np.random.randint(neg_size)
        while negative_table[idx] == verb or negative_table[idx] in neg_verbs:
            idx = np.random.randint(neg_size)
        neg_verbs.append(negative_table[idx])
    return neg_verbs


def load_verb_count(filename):
    count_list = []
    with codecs.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            count = int(line[1])
            if count < 90:
                break
            count_list.append(count)
    return count_list


def read_vocab_to_dict(filename):
    vocab = dict()
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        for idx, word in enumerate(f):
            word = word.lstrip().rstrip()
            vocab[word] = idx
    return vocab


def pad_sequence(sequence, pad_tok=0, max_length=None):
    """Pad batched dataset with shape = (batch_size, seq_length(various))
    :param sequence: input sequence
    :param pad_tok: padding token, default is 0
    :param max_length: max length of padded sequence, default is None
    :return: padded sequence
    """
    if max_length is None:
        max_length = max([len(seq) for seq in sequence])
    sequence_padded, seq_length = [], []
    for seq in sequence:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        seq_length.append(min(len(seq), max_length))
    return sequence_padded, seq_length


def build_batch_dataset(left_context, verbs, right_context):
    left_context, left_seq_len = pad_sequence(left_context)
    right_context, right_seq_len = pad_sequence(right_context)
    batch_size = len(verbs)
    return {"lc": left_context, "ll": left_seq_len, "rc": right_context, "rl": right_seq_len, "vb": verbs,
            "batch_size": batch_size}


def dataset_iterator(dataset_file, word_dict, verb_dict, batch_size):
    if dataset_file is None or not os.path.exists(dataset_file):
        raise IOError("Unable to find the dataset file: %s" % dataset_file)
    with codecs.open(dataset_file, mode="r", encoding="utf-8", errors="ignore") as f_dataset:
        left_context, verbs, right_context = [], [], []
        # for line in tqdm(f_dataset, total=int(2.663e8), desc="Read dataset"):
        for line in f_dataset:
            # split data
            l_c, vb, r_c, _ = line.strip().split("\t")
            # convert to indices
            l_c = [word_dict[word] if word in word_dict else word_dict[UNK] for word in l_c.split(" ")]
            vb = verb_dict[vb] if vb in verb_dict else verb_dict[UNK]
            r_c = [word_dict[word] if word in word_dict else word_dict[UNK] for word in r_c.split(" ")]
            # add to list
            left_context.append(l_c)
            verbs.append(vb)
            right_context.append(r_c)
            # yield batched dataset
            if len(left_context) == batch_size:
                yield build_batch_dataset(left_context, verbs, right_context)
                left_context, verbs, right_context = [], [], []
        if len(left_context) > 0:
            yield build_batch_dataset(left_context, verbs, right_context)


if __name__ == "__main__":
    # build dataset
    # neg_table = build_dataset_and_vocab("data")
    neg_table = build_negative_sampling_table("data/verb_count.txt")
    # negative sampling for verbs
    negative_sampling(os.path.join("data", "dataset.txt"), os.path.join("data", "dataset_neg.txt"), neg_table)
    # build verb vocabulary
