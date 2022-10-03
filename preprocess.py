# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   in the meantime, we do preprocess like capitalize the first character of a sentence or normalize digits
'''
import os
import json
from collections import Counter
import numpy as np
from tqdm import tqdm
import argparse
from io_utils import read_yaml, load_embedding_dict, save_pickle, read_pickle
from str_utils import normalize_tok
from vocab import Vocab
from actions import Actions
from nltk.parse import CoreNLPDependencyParser

joint_config = read_yaml('joint_config.yaml')
parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--seed', '-s', required=False, type=int, default=joint_config['random_seed'])
args = parser.parse_args()
joint_config['random_seed'] = args.seed
print('seed:', joint_config['random_seed'])
np.random.seed(joint_config['random_seed'])

data_config = read_yaml('data_config.yaml')
data_dir = data_config['data_dir']
cur_dataset_dir = data_config['cur_dataset_dir']
embedding_dir = data_config['embedding_dir']
embedding_file = data_config['embedding_file']
embedding_type = data_config['embedding_type']

normalize_digits = data_config['normalize_digits']
lower_case = data_config['lower_case']


ROLE_DICT = {'DS': 0, 'HOLDER': 1, 'TARGET': 2}
ROLE_DICT_REV = {v: k for k, v in ROLE_DICT.items()}


embedd_dict, embedd_dim = None, None
def read_embedding():
    global embedd_dict, embedd_dim
    embedd_dict, embedd_dim = load_embedding_dict(embedding_type,
                                                  os.path.join(embedding_dir, embedding_file),
                                                  normalize_digits=normalize_digits)
    print('Embedding type %s, file %s' % (embedding_type, embedding_file))


def func1(term_list, max_sent_len):
    res = []
    for d in term_list:
        if d not in res:
            if d[0] > max_sent_len:
                continue
            elif d[0] < max_sent_len < d[-1]:
                res.append([d[0], max_sent_len-1])
            else:
                res.append(d)
    return res


def get_dep(token_list, depparser):
    res = []
    parser_res = depparser.parse(token_list)
    for i in parser_res:
        temp = i.to_conll(4).strip().split('\n')
        for t in temp:
            res.append(t.split('\t'))
    return res


def get_new_ides(new_tokens, ori_tokens, ori_oht_token_list, ori_oht_ides_list, depparser):
    new_len = len(new_tokens)
    ori_len = len(ori_tokens)
    chazhi = new_len - ori_len
    new_ht_ides_list = []
    for oht_tokens, oht_ides in zip(ori_oht_token_list, ori_oht_ides_list):
        tokenized_tokens = list(depparser.tokenize(' '.join([normalize_tok(w) for w in oht_tokens])))
        try:
            new_ht_s = new_tokens.index(tokenized_tokens[0], oht_ides[0], oht_ides[0] + chazhi + 1)
        except ValueError as ve:
            print('index start error: ', ve)
            new_ht_s = new_tokens.index(''.join(tokenized_tokens[:2]), oht_ides[0], oht_ides[0] + chazhi + 1)
            print(''.join(tokenized_tokens[:2]), ' index correct.')

        try:
            new_ht_e = max(new_tokens.index(tokenized_tokens[-1], oht_ides[-1], oht_ides[-1] + chazhi + 1), new_ht_s+len(tokenized_tokens)-1)
        except ValueError as ve:
            print('index end error: ', ve)
            new_ht_e = max(new_tokens.index(''.join(tokenized_tokens[-2:]), oht_ides[-1], oht_ides[-1] + chazhi + 1), new_ht_s+len(tokenized_tokens)-2)
            print(''.join(tokenized_tokens[-2:]), ' index correct.')

        temp = [x for x in range(max(new_ht_s, new_ht_e-len(tokenized_tokens)+1), new_ht_e + 1)]
        new_ht_ides_list.append([temp[0], temp[-1]])
    return new_ht_ides_list


def build_vocab_inst_for_json_format(insts, depparser):
    token_list = []
    char_list = []
    role_list = []
    actions_list = []
    pos_list = []
    dep_types_list = ['selfloop']
    udog_types_list = ['HOLDER', 'TARGET', 'OPINION', 'ROLE']
    for inst in tqdm(insts, total=len(insts)):
        words = inst['sentences']
        ori_words = [normalize_tok(w) for w in words]
        new_words = []
        temp = get_dep(ori_words, depparser)

        for t in temp:
            token_list.append(t[0])
            new_words.append(t[0])
            char_list.extend(list(t[0]))
            pos_list.append(t[1])
            dep_types_list.append(t[3])

        dss = []
        relations = []
        holder = []
        target = []
        orl = inst['orl']
        for x in orl:
            ori_ides = x[2:4]
            new_ides = get_new_ides(new_words, ori_words,
                                    [words[x[2]: x[3]+1]],
                                    [ori_ides], depparser) if len(new_words) != len(ori_words) else [ori_ides]
            temp = get_new_ides(new_words, ori_words,
                                [words[x[0]: x[1]+1]],
                                [x[0:2]], depparser) if len(new_words) != len(ori_words) else [x[0:2]]
            if x[-1] == 'DSE':
                if new_ides not in dss:
                    dss.extend(new_ides)
            elif x[-1] == 'AGENT':
                if new_ides not in holder:
                    role_list.extend(['HOLDER'])
                    holder.extend(new_ides)
                    relations.append([temp[0][0], temp[0][-1], new_ides[0][0], new_ides[0][-1], 'HOLDER'])
            elif x[-1] == 'TARGET':
                if new_ides not in target:
                    role_list.extend(['TARGET'])
                    target.extend(new_ides)
                    relations.append([temp[0][0], temp[0][-1], new_ides[0][0], new_ides[0][-1], 'TARGET'])
            else:
                raise KeyError('annotation error, check {}'.format(' '.join(words)))
        term_dic = {'DSE{}'.format(i): x for i, x in enumerate(dss)}
        term_dic.update({'HOLDER{}'.format(i): x for i, x in enumerate(holder)})
        term_dic.update({'TARGET{}'.format(i): x for i, x in enumerate(target)})
        sorted_term_dic = dict(sorted(term_dic.items(), key=lambda x: (x[1], x[0])))
        term_start_end = []
        for k, v in sorted_term_dic.items():
            if k.startswith('D'):
                term_start_end.append([v[0], v[-1], 'DSE'])
            elif k.startswith('H'):
                term_start_end.append([v[0], v[-1], 'HOLDER'])
            elif k.startswith('T'):
                term_start_end.append([v[0], v[-1], 'TARGET'])
            else:
                print('unknown key type: {}: {}'.format(k, v))
        actions = Actions.make_oracle(new_words, sorted_term_dic, relations)
        actions_list.extend(actions)
    return token_list, char_list, actions_list, role_list, dep_types_list, pos_list, udog_types_list+dep_types_list


def build_vocab(train_list, dev_list, test_list):
    token_list = []
    char_list = []

    actions_list = []
    role_list = []
    pos_list = []
    dep_types_list = []
    udog_types_list = []

    depparser = CoreNLPDependencyParser(url='http://127.0.0.1:9000')

    train = build_vocab_inst_for_json_format(train_list, depparser)
    dev = build_vocab_inst_for_json_format(dev_list, depparser)
    test = build_vocab_inst_for_json_format(test_list, depparser)

    token_list.extend(train[0])
    token_list.extend(dev[0])
    token_list.extend(test[0])

    char_list.extend(train[1])
    char_list.extend(dev[1])
    char_list.extend(test[1])

    actions_list.extend(train[2])
    actions_list.extend(dev[2])
    actions_list.extend(test[2])

    role_list.extend(train[3])
    role_list.extend(dev[3])
    role_list.extend(test[3])

    dep_types_list.extend(train[4])
    dep_types_list.extend(dev[4])
    dep_types_list.extend(test[4])

    pos_list.extend(train[5])
    pos_list.extend(dev[5])
    pos_list.extend(test[5])

    udog_types_list.extend(train[6])
    udog_types_list.extend(dev[6])
    udog_types_list.extend(test[6])

    vocab_dir = data_config['vocab_dir']

    token_vocab_file = os.path.join(vocab_dir, data_config['token_vocab_file'])
    char_vocab_file = os.path.join(vocab_dir, data_config['char_vocab_file'])
    action_vocab_file = os.path.join(vocab_dir, data_config['action_vocab_file'])
    role_type_vocab_file = os.path.join(vocab_dir, data_config['role_type_vocab_file'])
    pos_vocab_file = os.path.join(vocab_dir, data_config['pos_vocab_file'])
    dep_type_vocab_file = os.path.join(vocab_dir, data_config['dep_type_vocab_file'])
    udog_type_vocab_file = os.path.join(vocab_dir, data_config['udog_type_vocab_file'])

    print('--------token_vocab---------------')
    token_vocab = Vocab()
    token_vocab.add_spec_toks(unk_tok=True, pad_tok=True)
    token_vocab.add_counter(Counter(token_list))
    token_vocab.save(token_vocab_file)
    print(token_vocab)

    print('--------char_vocab---------------')
    char_vocab = Vocab()
    char_vocab.add_spec_toks(unk_tok=True, pad_tok=True)
    char_vocab.add_counter(Counter(char_list))
    char_vocab.save(char_vocab_file)
    print(char_vocab)

    print('--------action_vocab---------------')
    action_vocab = Vocab()
    action_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    action_vocab.add_counter(Counter(actions_list))
    action_vocab.save(action_vocab_file)
    print(action_vocab)

    print('--------role_vocab---------------')
    role_vocab = Vocab()
    role_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    role_vocab.add_counter(Counter(role_list))
    role_vocab.save(role_type_vocab_file)
    print(role_vocab)

    print('--------pos_vocab---------------')
    pos_vocab = Vocab()
    pos_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    pos_vocab.add_counter(Counter(pos_list))
    pos_vocab.save(pos_vocab_file)
    print(pos_vocab)

    print('--------dep_type_vocab---------------')
    dep_type_vocab = Vocab()
    dep_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    dep_type_vocab.add_counter(Counter(dep_types_list))
    dep_type_vocab.save(dep_type_vocab_file)
    print(dep_type_vocab)

    print('--------udog_type_vocab---------------')
    udog_type_vocab = Vocab()
    udog_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    udog_type_vocab.add_counter(Counter(udog_types_list))
    udog_type_vocab.save(udog_type_vocab_file)
    print(udog_type_vocab)


def get_graph(seq_len, dep_head, dep_rel_indices, dep_type_vocab):
    ret = [[0] * seq_len for _ in range(seq_len)]
    for i, item in enumerate(dep_head):
        if int(item) > seq_len-1:
            continue
        elif int(item) == -1:
            ret[i][i] = dep_type_vocab.get_index('selfloop')
        else:
            try:
                ret[i][int(item)] = dep_rel_indices[i]
                ret[int(item)][i] = dep_rel_indices[i]
                ret[i][i] = dep_type_vocab.get_index('selfloop')
            except:
                print('xxxxxx')
                raise IndexError(seq_len, i, item)
    return ret


def construct_instance_for_json_format(inst_list, token_vocab, char_vocab,
                                       action_vocab, pos_vocab, dep_type_vocab, udog_type_vocab, depparser):
    word_num = 0
    processed_inst_list = []
    for inst in tqdm(inst_list, total=len(inst_list)):
        new_inst = {}

        words = inst['sentences']

        words_processed = []
        word_indices = []
        char_indices = []
        pos, pos_indices = [], []
        dep_rel, dep_rel_indices, dep_head = [], [], []
        ori_words = [normalize_tok(w) for w in words]
        new_words = []
        temp = get_dep(ori_words, depparser)

        for t in temp:
            words_processed.append(t[0])
            word_idx = token_vocab.get_index(t[0])
            word_indices.append(word_idx)
            char_indices.append([char_vocab.get_index(c) for c in t[0]])

            new_words.append(t[0])

            pos.append(t[1])
            pos_indices.append(pos_vocab.get_index(t[1]))
            dep_head.append(int(t[2]) - 1)
            dep_rel.append(t[3])
            dep_rel_indices.append(dep_type_vocab.get_index(t[3]))
        if len(new_words) > joint_config['max_sent_len']:
            continue
        sent_len = min(len(word_indices), joint_config['max_sent_len'])
        assert len(dep_head) == len(new_words) and len(dep_head) <= joint_config['max_sent_len']
        graph = get_graph(sent_len, dep_head[:joint_config['max_sent_len']],
                          dep_rel_indices[:joint_config['max_sent_len']], udog_type_vocab)
        new_inst['graph'] = graph
        neibour_index = []
        neibour_type = []
        for i in range(sent_len):
            t_1 = []
            t_2 = []
            for j in range(sent_len):
                if graph[i][j] != 0:
                    t_1.append(j)
                    t_2.append(graph[i][j])
            assert len(t_1) == len(t_2)
            neibour_index.append(t_1.copy())
            neibour_type.append(t_2.copy())
        new_inst['neibour_index'] = neibour_index
        new_inst['neibour_type'] = neibour_type

        orginal_len = len(word_indices)

        if orginal_len < joint_config['max_sent_len']:
            new_inst['words'] = words_processed
            new_inst['word_indices'] = word_indices + [token_vocab.get_index('*PAD*')] * (
                    joint_config['max_sent_len'] - orginal_len)
            new_inst['char_indices'] = char_indices + [[char_vocab.get_index(x) for x in '*PAD*']] * (
                    joint_config['max_sent_len'] - orginal_len)
            new_inst['mask'] = [1] * orginal_len + [0] * (joint_config['max_sent_len'] - orginal_len)
            new_inst['pos'] = pos
            new_inst['pos_indices'] = pos_indices + [pos_vocab.get_index('*NULL*')] * (
                    joint_config['max_sent_len'] - orginal_len)
            new_inst['dep_rel'] = dep_rel
            new_inst['dep_rel_indices'] = dep_rel_indices + [dep_type_vocab.get_index('*NULL*')] * (
                    joint_config['max_sent_len'] - orginal_len)
            new_inst['dep_head'] = dep_head + [0] * (joint_config['max_sent_len'] - orginal_len)
        else:
            new_inst['words'] = words_processed[:joint_config['max_sent_len']]
            new_inst['word_indices'] = word_indices[:joint_config['max_sent_len']]
            new_inst['char_indices'] = char_indices[:joint_config['max_sent_len']]
            new_inst['mask'] = [1] * joint_config['max_sent_len']
            new_inst['pos'] = pos[:joint_config['max_sent_len']]
            new_inst['pos_indices'] = pos_indices[:joint_config['max_sent_len']]
            new_inst['dep_rel'] = dep_rel[:joint_config['max_sent_len']]
            new_inst['dep_rel_indices'] = dep_rel_indices[:joint_config['max_sent_len']]
            new_inst['dep_head'] = dep_head[:joint_config['max_sent_len']]


        dss = []
        holder = []
        target = []
        relations = []
        orl = inst['orl']
        for x in orl:
            ori_ides = x[2:4]
            new_ides = get_new_ides(new_words, ori_words,
                                    [words[x[2]: x[3] + 1]],
                                    [ori_ides], depparser) if len(new_words) != len(ori_words) else [ori_ides]
            new_ides = func1(new_ides, sent_len)
            temp = get_new_ides(new_words, ori_words,
                                [words[x[0]: x[1] + 1]],
                                [x[0:2]], depparser) if len(new_words) != len(ori_words) else [x[0:2]]
            temp = func1(temp, sent_len)
            if x[-1] == 'DSE':
                if len(new_ides) > 0 and new_ides[0] not in dss:
                    dss.extend(new_ides)
            elif x[-1] == 'AGENT':
                if len(new_ides) > 0 and new_ides[0] not in holder:
                    holder.extend(new_ides)
                    if len(temp) > 0:
                        relations.append([temp[0][0], temp[0][-1], new_ides[0][0], new_ides[0][-1], 'HOLDER'])
            elif x[-1] == 'TARGET':
                if len(new_ides) > 0 and new_ides[0] not in target:
                    target.extend(new_ides)
                    if len(temp) > 0:
                        relations.append([temp[0][0], temp[0][-1], new_ides[0][0], new_ides[0][-1], 'TARGET'])
            else:
                raise KeyError('annotation error, check {}'.format(' '.join(words)))

        new_inst['dss'] = dss
        new_inst['holders'] = holder
        new_inst['targets'] = target
        new_inst['relations'] = relations
        new_inst['ht'] = holder + target

        term_dic = {'DSE{}'.format(i): x for i, x in enumerate(dss)}
        term_dic.update({'HOLDER{}'.format(i): x for i, x in enumerate(holder)})
        term_dic.update({'TARGET{}'.format(i): x for i, x in enumerate(target)})
        sorted_term_dic = dict(sorted(term_dic.items(), key=lambda x: (x[1], x[0])))
        term_start_end = []
        for k, v in sorted_term_dic.items():
            if k.startswith('D'):
                term_start_end.append([v[0], v[-1], 'DSE'])
            elif k.startswith('H'):
                term_start_end.append([v[0], v[-1], 'HOLDER'])
            elif k.startswith('T'):
                term_start_end.append([v[0], v[-1], 'TARGET'])
            else:
                print('unknown key type: {}: {}'.format(k, v))
        actions = Actions.make_oracle(new_words, sorted_term_dic, relations)
        new_inst['actions'] = actions
        new_inst['action_indices'] = [action_vocab.get_index(act) for act in actions]
        new_inst['sent_range'] = list(range(word_num, word_num + len(words_processed)))
        word_num += len(words_processed)
        processed_inst_list.append(new_inst.copy())
    return processed_inst_list


def pickle_data(train_list, dev_list, test_list, fold_idx):
    vocab_dir = data_config['vocab_dir']

    token_vocab_file = os.path.join(vocab_dir, data_config['token_vocab_file'])
    char_vocab_file = os.path.join(vocab_dir, data_config['char_vocab_file'])
    action_vocab_file = os.path.join(vocab_dir, data_config['action_vocab_file'])
    role_type_vocab_file = os.path.join(vocab_dir, data_config['role_type_vocab_file'])
    pos_vocab_file = os.path.join(vocab_dir, data_config['pos_vocab_file'])
    dep_type_vocab_file = os.path.join(vocab_dir, data_config['dep_type_vocab_file'])
    udog_type_vocab_file = os.path.join(vocab_dir, data_config['udog_type_vocab_file'])

    token_vocab = Vocab.load(token_vocab_file)
    char_vocab = Vocab.load(char_vocab_file)
    role_type_vocab = Vocab.load(role_type_vocab_file)
    action_vocab = Vocab.load(action_vocab_file)
    pos_vocab = Vocab.load(pos_vocab_file)
    dep_type_vocab = Vocab.load(dep_type_vocab_file)
    udog_type_vocab = Vocab.load(udog_type_vocab_file)

    depparser = CoreNLPDependencyParser(url='http://127.0.0.1:9000')

    processed_train = construct_instance_for_json_format(train_list, token_vocab, char_vocab, action_vocab,
                                                         pos_vocab, dep_type_vocab, udog_type_vocab, depparser)
    processed_dev = construct_instance_for_json_format(dev_list, token_vocab, char_vocab, action_vocab,
                                                       pos_vocab, dep_type_vocab, udog_type_vocab, depparser)
    processed_test = construct_instance_for_json_format(test_list, token_vocab, char_vocab, action_vocab,
                                                        pos_vocab, dep_type_vocab, udog_type_vocab, depparser)

    pickle_dir = data_config['pickle_dir']
    inst_pl_file = data_config['inst_pl_file']
    save_path = os.path.join(pickle_dir, str(fold_idx), inst_pl_file)

    print('Saving pickle to ', save_path)
    print('Saving sent size Train: %d, Dev: %d, Test:%d' % (
        len(processed_train), len(processed_dev), len(processed_test)))
    save_pickle(save_path, [processed_train, processed_dev, processed_test, token_vocab, char_vocab,
                            action_vocab, role_type_vocab, pos_vocab, dep_type_vocab, udog_type_vocab])


def load_embedding():
    pickle_dir = data_config['pickle_dir']
    vec_npy_file = data_config['vec_npy']

    vocab_dir = data_config['vocab_dir']
    token_vocab_file = os.path.join(vocab_dir, data_config['token_vocab_file'])
    token_vocab = Vocab.load(token_vocab_file)

    scale = np.sqrt(3.0 / embedd_dim)
    vocab_dict = token_vocab.tok2idx
    table = np.empty([len(vocab_dict), embedd_dim], dtype=np.float32)
    oov = 0
    for word, index in vocab_dict.items():
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding

    np.save(os.path.join(pickle_dir, vec_npy_file), table)
    print('pretrained embedding oov: %d' % oov)
    print()


def drop_list(x, max_length=joint_config['max_target_length']):
    drop_flag = [0] * len(x)

    for m_i in range(len(x)):
        for n_i in range(len(x)):
            if m_i != n_i:
                if x[m_i] == x[n_i]:
                    drop_flag[min([m_i, n_i])] = 1
                elif x[m_i][0] <= x[n_i][0] and x[m_i][-1] >= x[n_i][-1]:
                    drop_flag[m_i] = 1

    res = []
    for m_i in range(len(x)):
        if drop_flag[m_i] == 0 and len(x[m_i]) < max_length:
            res.append(x[m_i])
    return res


def count_info_for_json_format(inst_list):
    h_t_olp = 0.0
    o_h_olp = 0.0
    o_t_olp = 0.0
    t_t_olp = 0.0
    o_o_olp = 0.0
    h_h_olp = 0.0

    dse_len = []
    role_len = []
    h_len = []
    t_len = []

    t_num, h_num, all_num = [], [], []

    overlap_data = []

    for inst in tqdm(inst_list, total=len(inst_list)):
        dss = []
        holder = []
        target = []
        all_term = []
        orl = inst['orl']
        for x in orl:
            ori_ides = x[2:4]
            if ori_ides not in all_term:
                all_term.append(ori_ides)
            if x[-1] == 'DSE':
                dse_len.append(x[3]-x[2]+1)
                if ori_ides not in dss:
                    dss.append(ori_ides)
            elif x[-1] == 'AGENT':
                role_len.append(x[3]-x[2]+1)
                h_len.append(x[3]-x[2]+1)
                if ori_ides not in holder:
                    holder.append(ori_ides)
                if not (x[1] <= x[2] or x[3] <= x[0]):
                    o_h_olp += 1
            elif x[-1] == 'TARGET':
                role_len.append(x[3] - x[2] + 1)
                t_len.append(x[3] - x[2] + 1)
                if ori_ides not in target:
                    target.append(ori_ides)
                if not (x[1] <= x[2] or x[3] <= x[0]):
                    o_t_olp += 1
            else:
                raise KeyError('annotation error, check {}'.format(' '.join(inst['sentences'])))
        t_num.append(len(target))
        h_num.append(len(holder))
        all_num.append(len(holder)+len(target))
        for i in range(len(all_term)):
            for j in range(i + 1, len(all_term)):
                if not (all_term[i][-1] <= all_term[j][0] or all_term[j][-1] <= all_term[i][0]):
                    overlap_data.append(inst)

        if len(dss) > 1:
            for i in range(len(dss)):
                for j in range(i+1, len(dss)):
                    if not (dss[i][-1] <= dss[j][0] or dss[j][-1] <= dss[i][0]):
                        o_o_olp += 1

        if len(holder) > 1:
            for i in range(len(holder)):
                for j in range(i + 1, len(holder)):
                    if not (holder[i][-1] <= holder[j][0] or holder[j][-1] <= holder[i][0]):
                        h_h_olp += 1
                for k in range(len(target)):
                    if not (holder[i][-1] <= target[k][0] or target[k][-1] <= holder[i][0]):
                        h_t_olp += 1

        if len(target) > 1:
            for i in range(len(target)):
                for j in range(i + 1, len(target)):
                    if not (target[i][-1] <= target[j][0] or target[j][-1] <= target[i][0]):
                        t_t_olp += 1

    # print('----------- number count -----------------')
    print(Counter(t_num).most_common())
    print(Counter(h_num).most_common())
    print(Counter(all_num).most_common())
    print('----------------------------')
    print(len(role_len))
    # print(Counter(role_len).most_common(8))
    temp = 0
    for x in Counter(role_len).most_common(8):
        temp += x[1]
    print((len(role_len)-temp)/len(role_len))

    print(len(h_len))
    # print(Counter(h_len).most_common(8))
    temp = 0
    for x in Counter(h_len).most_common(8):
        temp += x[1]
    print((len(h_len) - temp) / len(h_len))

    print(len(t_len))
    # print(Counter(t_len).most_common(8))
    temp = 0
    for x in Counter(t_len).most_common(8):
        temp += x[1]
    print((len(t_len) - temp) / len(t_len))

    print(len(dse_len))
    # print(Counter(dse_len).most_common(8))
    temp = 0
    for x in Counter(dse_len).most_common(8):
        temp += x[1]
    print((len(dse_len) - temp) / len(dse_len))
    print('----------------------------')


if __name__ == '__main__':
    # train_list = []
    # dev_list = []
    # test_list = []
    # for i in range(5):
    #     with open(os.path.join(data_config['cur_dataset_dir'], str(i), 'train.json')) as f:
    #         train_list.extend([json.loads(line) for line in f])
    #     with open(os.path.join(data_config['cur_dataset_dir'], str(i), 'dev.json')) as f:
    #         dev_list.extend([json.loads(line) for line in f])
    #     with open(os.path.join(data_config['cur_dataset_dir'], str(i), 'test.json')) as f:
    #         test_list.extend([json.loads(line) for line in f])
    # build_vocab(train_list, dev_list, test_list)
    read_embedding()
    load_embedding()

    for i in range(5):
        with open(os.path.join(data_config['cur_dataset_dir'], str(i), 'train.json')) as f:
            train_list = [json.loads(line) for line in f]
        with open(os.path.join(data_config['cur_dataset_dir'], str(i), 'dev.json')) as f:
            dev_list = [json.loads(line) for line in f]
        with open(os.path.join(data_config['cur_dataset_dir'], str(i), 'test.json')) as f:
            test_list = [json.loads(line) for line in f]
        pickle_data(train_list[:100], dev_list[:100], test_list[:100], i)
        # print('------ fold {} -------'.format(i))
        # print('train')
        # count_info_for_json_format(train_list)
        # print('dev')
        # count_info_for_json_format(dev_list)
        # print('test')
        # count_info_for_json_format(test_list)

    print('xxx')
