import os
import math
import numpy as np

from synptrtrans.model import *


from io_utils import read_pickle, write_lines, read_lines, write_texts
from vocab import Vocab
from tqdm import tqdm


class Trainer(object):

    def __init__(self, fold_idx):
        logger.info('Loading data...')
        pickle_dir = data_config['pickle_dir']
        vec_npy_file = data_config['vec_npy']
        inst_pl_file = data_config['inst_pl_file']
        data_file_path = os.path.join(pickle_dir, str(fold_idx), inst_pl_file)
        vec_file_path = os.path.join(pickle_dir, vec_npy_file)

        self.fold_idx = fold_idx

        self.train_list, self.dev_list, self.test_list, self.token_vocab, self.char_vocab, \
        self.action_vocab, self.opi_role_vocab, self.pos_vocab, self.dep_type_vocab, self.udog_type_vocab = read_pickle(data_file_path)

        logger.info('cur_dataset_dir: %s' % data_config['cur_dataset_dir'])
        logger.info(
            'Sent size train: %d, dev: %d, test:%d' % (len(self.train_list), len(self.dev_list), len(self.test_list)))

        logger.info('Loading pretrained from: %s' % vec_file_path)
        pretrained_vec = np.load(vec_file_path)

        self.unk_idx = self.token_vocab[Vocab.UNK]
        joint_config['n_chars'] = len(self.char_vocab)
        self.trans_model = SynPtrTrans(self.token_vocab.get_vocab_size(),
                                       self.action_vocab,
                                       self.opi_role_vocab,
                                       self.dep_type_vocab,
                                       self.pos_vocab,
                                       pretrained_vec=pretrained_vec,
                                       udog_type_vocab=self.udog_type_vocab)
        logger.info("Model:%s" % type(self.trans_model))

        self.orl_eval = OrlEval()

    def unk_replace_singleton(self, unk_idx, unk_ratio, words):
        noise = words[:]
        bernoulli = np.random.binomial(n=1, p=unk_ratio, size=len(words))
        for i, idx in enumerate(words):
            if self.token_vocab.is_singleton(idx) and bernoulli[i] == 1:
                noise[i] = unk_idx
        return noise

    def iter_batch(self, inst_list, shuffle=True):
        batch_size = joint_config['batch_size']
        if shuffle:
            random.shuffle(inst_list)
        inst_len = len(inst_list)
        plus_n = 0 if (inst_len % batch_size) == 0 else 1
        num_batch = (len(inst_list) // batch_size) + plus_n

        start = 0
        for i in range(num_batch):
            batch_inst = inst_list[start: start + batch_size]
            start += batch_size
            yield batch_inst

    def train_batch(self):
        loss_all = 0.
        batch_num = 0
        # i = 1
        # batch_loss = None
        for batch_inst in tqdm(self.iter_batch(self.train_list, shuffle=True), total=len(self.train_list)):
            dy.renew_cg()
            loss_minibatch = []
            for inst in batch_inst:
                words = inst['word_indices']
                if joint_config['unk_ratio'] > 0:
                    words = self.unk_replace_singleton(self.unk_idx, joint_config['unk_ratio'], words)
                loss_rep = self.trans_model(words, inst['char_indices'],
                                            inst['action_indices'], inst['actions'],
                                            inst['sent_range'],
                                            inst['pos_indices'], inst['dep_rel_indices'], inst['dep_head'],
                                            len(inst['words']), inst['dss'], inst['relations'],
                                            neibour_index=inst['neibour_index'], neibour_type=inst['neibour_type'])

                loss_minibatch.append(loss_rep)

            batch_loss = dy.esum(loss_minibatch) / len(loss_minibatch)
            loss_all += batch_loss.value()
            batch_loss.backward()
            self.trans_model.update()
            batch_num += 1

        logger.info('loss %.5f ' % (loss_all / float(batch_num)))

    def eval(self, inst_list, write_file_path=None, save_file_path=None, is_write_ent=False, mtype='dev'):
        self.orl_eval.reset()
        sent_num_eval = 0

        ent_lines = []
        results = []
        for idx, inst in tqdm(enumerate(inst_list), total=len(inst_list)):
            if idx == 1674 or idx == 1675:
                continue
            pred_opinions, pred_holders, pred_targets, pred_pairs = self.trans_model.decode(
                inst['word_indices'], inst['char_indices'], inst['action_indices'], inst['actions'],
                inst['sent_range'], inst['pos_indices'], inst['dep_rel_indices'], inst['dep_head'], mtype=mtype,
                sent_len=len(inst['words']), dss=inst['dss'], relations=inst['relations'],
                neibour_index=inst['neibour_index'], neibour_type=inst['neibour_type'])

            self.orl_eval.update(pred_opinions, inst['dss'],  # list
                                 pred_holders, inst['holders'],  # list
                                 pred_targets, inst['targets'],  # triplet
                                 eval_arg=True, words=inst['words'])

            self.orl_eval.update_pair(pred_pairs, inst['relations'])
            results.append({'sentence': inst['words'], 'gold_orl': inst['relations'], 'sys_orl': pred_pairs})

            ent_line = str(sent_num_eval) + ' \n'
            ent_line += 'gold opinions:\t ' + ' ## '.join([str(i) for i in inst['dss']]) + '\n'
            ent_line += 'pred opinions:\t ' + ' ## '.join([str(i) for i in pred_opinions]) + '\n'
            ent_line += 'gold holders:\t ' + ' ## '.join([str(i) for i in inst['holders']]) + '\n'
            ent_line += 'pred holders:\t ' + ' ## '.join([str(i) for i in pred_holders]) + '\n'
            ent_line += 'gold targets:\t ' + ' ## '.join([str(i) for i in inst['targets']]) + '\n'
            ent_line += 'pred targets:\t ' + ' ## '.join([str(i) for i in pred_targets]) + '\n'
            ent_line += ' \n\n'
            ent_lines.append(ent_line)

            sent_num_eval += 1

        if write_file_path is not None and is_write_ent:
            write_texts(write_file_path, ent_lines)

        import json
        if save_file_path is not None:
            m_res = []
            for res in results:
                m_res.append(json.dumps(res)+"\n")
            with open(save_file_path, 'w', encoding='utf-8') as f:
                f.writelines(m_res)

    def train(self, save_model=True):
        logger.info(joint_config['msg_info'])
        best_f1_prd, best_f1_f_pair = 0, 0
        best_epoch = 0

        adjust_lr = False
        stop_patience = joint_config['patience']
        stop_count = 0
        eval_best_arg = True  # for other task than event
        t_cur = 1
        t_i = 4
        t_mul = 2
        lr_max, lr_min = joint_config['init_lr'], joint_config['minimum_lr']
        for epoch in range(joint_config['num_epochs']):
            anneal_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_cur / t_i))
            self.trans_model.set_lr(anneal_lr)

            logger.info('--------------------------------------')
            logger.info('Epoch : %d' % (epoch+1))
            logger.info('LR : %.5f' % self.trans_model.get_lr())

            self.train_batch()

            write_file_path = os.path.join(data_config['eval_dir'], str(self.fold_idx),
                                           data_config['write_eval_file_dev'])
            self.eval(self.dev_list, write_file_path=write_file_path, is_write_ent=True, mtype='dev')
            (p_opinions, r_opinions, f_opinions), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets)\
                = self.orl_eval.report()

            logger.info('              P       R      F1')
            logger.info('opinions  %.5f\t%.5f\t%.5f' % (p_opinions, r_opinions, f_opinions))
            logger.info('holders   %.5f\t%.5f\t%.5f' % (p_holders, r_holders, f_holders))
            logger.info('targets   %.5f\t%.5f\t%.5f' % (p_targets, r_targets, f_targets))

            if t_cur == t_i:
                t_cur = 0
                t_i *= t_mul

            t_cur += 1

            if not eval_best_arg:
                continue
            if (f_opinions+f_holders+f_targets) > best_f1_f_pair:
                best_f1_prd = f_opinions+f_holders+f_targets
                best_f1_f_pair = f_opinions+f_holders+f_targets
                best_epoch = epoch

                stop_count = 0

                if save_model:
                    save_model_path = os.path.join(data_config['pickle_dir'], str(self.fold_idx),
                                                   data_config['model_save_file'])
                    logger.info('Saving model %s' % save_model_path)
                    self.trans_model.save_model(save_model_path)

            else:
                stop_count += 1
                if stop_count >= stop_patience:
                    logger.info('Stop training, Arg performance did not improved for %d epochs' % stop_count)
                    break

                if adjust_lr:
                    self.trans_model.decay_lr(joint_config['decay_lr'])
                    logger.info('@@@@  Adjusting LR: %.5f  @@@@@@' % self.trans_model.get_lr())

                if self.trans_model.get_lr() < joint_config['minimum_lr']:
                    adjust_lr = False

            best_msg = '*****Best epoch: %d prd and pair F:%.5f, F:%.5f ******' % (best_epoch+1,
                                                                                   best_f1_prd,
                                                                                   best_f1_f_pair)
            logger.info(best_msg)

        return best_msg, best_f1_f_pair

    def test(self, fname):
        self.trans_model.load_model(fname)
        write_file_path = os.path.join(data_config['eval_dir'], str(self.fold_idx), data_config['write_eval_file_test'])
        save_file_path = os.path.join(data_config['eval_dir'], str(self.fold_idx), data_config['save_eval_file_test'])
        self.eval(self.test_list, write_file_path=write_file_path, save_file_path=save_file_path,
                  is_write_ent=True, mtype='test')
        (p_opinions, r_opinions, f_opinions), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets) \
            = self.orl_eval.report()
        logger.info("--------------Hard-----------------")
        logger.info('              P       R      F1')
        logger.info('opinions  %.5f\t%.5f\t%.5f' % (p_opinions, r_opinions, f_opinions))
        logger.info('holders   %.5f\t%.5f\t%.5f' % (p_holders, r_holders, f_holders))
        logger.info('targets   %.5f\t%.5f\t%.5f' % (p_targets, r_targets, f_targets))

        (p_opinions, r_opinions, f_opinions), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets) \
            = self.orl_eval.report_pair()
        logger.info("--------------Hard-----------------")
        logger.info('              P       R      F1')
        logger.info('opinions  %.5f\t%.5f\t%.5f' % (p_opinions, r_opinions, f_opinions))
        logger.info('holders   %.5f\t%.5f\t%.5f' % (p_holders, r_holders, f_holders))
        logger.info('targets   %.5f\t%.5f\t%.5f' % (p_targets, r_targets, f_targets))



if __name__ == '__main__':

    random.seed(joint_config['random_seed'])
    np.random.seed(joint_config['random_seed'])
    for i in range(5):
        trainer = Trainer(i)
        trainer.train(save_model=True)

        logger.info('---------------fold %d Test Results---------------' % i)
        ckp_path = os.path.join(data_config['pickle_dir'], str(i), data_config['model_save_file'])
        trainer.test(ckp_path)
