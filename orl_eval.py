
from vocab import Vocab

def to_set(input):
    out_set = set()
    out_type_set = set()
    for x in input:
        out_set.add(tuple(x[:-1]))
        out_type_set.add(tuple(x))

    return out_set, out_type_set


class OrlEval(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_opinions = 0.
        self.num_pre_opinions = 0.
        self.num_gold_opinions = 0.

        self.correct_holders = 0.
        self.num_pre_holders = 0.
        self.num_gold_holders = 0.

        self.correct_targets = 0.
        self.num_pre_targets = 0.
        self.num_gold_targets = 0.

        self.gold_ds_t_num = 0.
        self.pred_ds_t_num = 0.
        self.correct_ds_t_num = 0.

        self.gold_ds_h_num = 0.
        self.pred_ds_h_num = 0.
        self.correct_ds_h_num = 0.

        self.correct_opinions_binary = 0.
        self.correct_opinions_proportional = 0.
        self.correct_holders_binary = 0.
        self.correct_holders_proportional = 0.
        self.correct_targets_binary = 0.
        self.correct_targets_proportional = 0.

    def update(self, pred_opinions, gold_opinions,
               pred_holders, gold_holders,
               pred_targets, gold_targets, eval_arg=True, words=None):

        self.num_pre_opinions += len(pred_opinions)
        self.num_gold_opinions += len(gold_opinions)

        self.num_pre_holders += len(pred_holders)
        self.num_gold_holders += len(gold_holders)

        self.num_pre_targets += len(pred_targets)
        self.num_gold_targets += len(gold_targets)
        for i in gold_opinions:
            for j in pred_opinions:
                if i[0] == j[0] and i[-1] == j[-1]:
                    self.correct_opinions += 1

        for i in gold_holders:
            for j in pred_holders:
                if i[0] == j[0] and i[-1] == j[-1]:
                    self.correct_holders += 1

        for i in gold_targets:
            for j in pred_targets:
                if i[0] == j[0] and i[-1] == j[-1]:
                    self.correct_targets += 1

    def report(self):
        p_opinions = self.correct_opinions / (self.num_pre_opinions + 1e-18)
        r_opinions = self.correct_opinions / (self.num_gold_opinions + 1e-18)
        f_opinions = 2 * p_opinions * r_opinions / (p_opinions + r_opinions + 1e-18)

        p_holders = self.correct_holders / (self.num_pre_holders + 1e-18)
        r_holders = self.correct_holders / (self.num_gold_holders + 1e-18)
        f_holders = 2 * p_holders * r_holders / (p_holders + r_holders + 1e-18)

        p_targets = self.correct_targets / (self.num_pre_targets + 1e-18)
        r_targets = self.correct_targets / (self.num_gold_targets + 1e-18)
        f_targets = 2 * p_targets * r_targets / (p_targets + r_targets + 1e-18)


        return (p_opinions, r_opinions, f_opinions), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets)

    def report_binary(self):
        p_opinions = self.correct_opinions_binary / (self.num_pre_opinions + 1e-18)
        r_opinions = self.correct_opinions_binary / (self.num_gold_opinions + 1e-18)
        f_opinions = 2 * p_opinions * r_opinions / (p_opinions + r_opinions + 1e-18)

        p_holders = self.correct_holders_binary / (self.num_pre_holders + 1e-18)
        r_holders = self.correct_holders_binary / (self.num_gold_holders + 1e-18)
        f_holders = 2 * p_holders * r_holders / (p_holders + r_holders + 1e-18)

        p_targets = self.correct_targets_binary / (self.num_pre_targets + 1e-18)
        r_targets = self.correct_targets_binary / (self.num_gold_targets + 1e-18)
        f_targets = 2 * p_targets * r_targets / (p_targets + r_targets + 1e-18)

        return (p_opinions, r_opinions, f_opinions), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets)

    def report_proportional(self):
        p_opinions = self.correct_opinions_proportional / (self.num_pre_opinions + 1e-18)
        r_opinions = self.correct_opinions_proportional / (self.num_gold_opinions + 1e-18)
        f_opinions = 2 * p_opinions * r_opinions / (p_opinions + r_opinions + 1e-18)

        p_holders = self.correct_holders_proportional / (self.num_pre_holders + 1e-18)
        r_holders = self.correct_holders_proportional / (self.num_gold_holders + 1e-18)
        f_holders = 2 * p_holders * r_holders / (p_holders + r_holders + 1e-18)

        p_targets = self.correct_targets_proportional / (self.num_pre_targets + 1e-18)
        r_targets = self.correct_targets_proportional / (self.num_gold_targets + 1e-18)
        f_targets = 2 * p_targets * r_targets / (p_targets + r_targets + 1e-18)

        return (p_opinions, r_opinions, f_opinions), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets)

    def get_coref_ent(self, g_ent_typed):
        ent_ref_dict = {}
        for ent1 in g_ent_typed:
            start1, end1, ent_type1, ent_ref1 = ent1
            coref_ents = []
            ent_ref_dict[(start1, end1)] = coref_ents
            for ent2 in g_ent_typed:
                start2, end2, ent_type2, ent_ref2 = ent2
                if ent_ref1 == ent_ref2:
                    coref_ents.append((start2, end2))
        return ent_ref_dict

    def split_prob(self, pred_args):
        sp_args, probs = [], []
        for arg in pred_args:
            sp_args.append(arg[:-1])
            probs.append(arg[-1])
        return sp_args, probs

    def update_1(self, pred_opinions, pred_holders, pred_targets, ds_ht_pairs):
        gold_opinions = []
        gold_holders = []
        gold_targets = []
        for ds_ht in ds_ht_pairs:
            gold_opinions.extend(ds_ht['ds'])
            gold_holders.extend(ds_ht['holder'])
            gold_targets.extend(ds_ht['target'])
        self.update(pred_opinions, gold_opinions, pred_holders, gold_holders, pred_targets, gold_targets)

    def count_binary_proportional(self, pred, gold):
        binary = 0.0
        proportional = 0
        # num_gold = 0
        # num_pred = 0
        # intersect_binary_temp = 0.0
        # intersect_proportional_temp = 0.0
        for entity_pred in pred:
            flag = 0
            entity_pred = [x for x in range(entity_pred[0], entity_pred[-1]+1)]
            for entity_gold in gold:
                if (len(list(set(entity_pred) & set(entity_gold))) >= 1) and (flag == 0):
                    binary += 1
                    proportional += len(list(set(entity_pred) & set(entity_gold))) / float(
                        len(entity_gold))
                    flag = 1
        # binary = intersect_binary_temp
        # proportional = intersect_proportional_temp
        num_gold = len(gold)
        num_pred = len(pred)

        return binary, proportional, num_gold, num_pred

    def bp(self,
           pred_opinions, gold_opinions,
           pred_holders, gold_holders,
           pred_targets, gold_targets):
        res_opi = self.count_binary_proportional(pred_opinions, gold_opinions)
        self.correct_opinions_binary += res_opi[0]
        self.correct_opinions_proportional += res_opi[1]
        self.num_gold_opinions += res_opi[2]
        self.num_pre_opinions += res_opi[3]

        res_h = self.count_binary_proportional(pred_holders, gold_holders)
        self.correct_holders_binary += res_h[0]
        self.correct_holders_proportional += res_h[1]
        self.num_gold_holders += res_h[2]
        self.num_pre_holders += res_h[3]

        res_t = self.count_binary_proportional(pred_targets, gold_targets)
        self.correct_targets_binary += res_t[0]
        self.correct_targets_proportional += res_t[1]
        self.num_gold_targets += res_t[2]
        self.num_pre_targets += res_t[3]

    def bp_1(self, pred_opinions, pred_holders, pred_targets, ds_ht_pairs):
        gold_opinions = []
        gold_holders = []
        gold_targets = []
        for ds_ht in ds_ht_pairs:
            gold_opinions.extend(ds_ht['ds'])
            gold_holders.extend(ds_ht['holder'])
            gold_targets.extend(ds_ht['target'])
        self.bp(pred_opinions, gold_opinions, pred_holders, gold_holders, pred_targets, gold_targets)

    def count_binary_proportional_for_pair(self, pred, gold):
        binary = 0.0
        proportional = 0
        for entity_pred in pred:
            flag = 0
            entity_pred = [x for x in range(entity_pred[0], entity_pred[-1]+1)]
            for entity_gold in gold:
                if (len(list(set(entity_pred) & set(entity_gold))) >= 1) and (flag == 0):
                    binary += 1
                    proportional += len(list(set(entity_pred) & set(entity_gold))) / float(
                        len(entity_gold))
                    flag = 1
        num_gold = len(gold)
        num_pred = len(pred)

        return binary, proportional, num_gold, num_pred

    def update_pair(self, pred_pairs, gold_pairs):

        gold_ds_h, gold_ds_t = [], []
        pred_ds_h, pred_ds_t = [], []
        for x in gold_pairs:
            if x[4] == 'HOLDER':
                gold_ds_h.append(x[:4])
            else:
                gold_ds_t.append(x[:4])
        for x in pred_pairs:
            if x[2] == 'HOLDER':
                pred_ds_h.append([x[0], x[1]])
            else:
                pred_ds_t.append([x[0], x[1]])

        self.gold_ds_h_num += len(gold_ds_h)
        self.gold_ds_t_num += len(gold_ds_t)
        self.pred_ds_h_num += len(pred_ds_h)
        self.pred_ds_t_num += len(pred_ds_t)
        for m in gold_ds_h:
            for n in pred_ds_h:
                if m[0] == n[0][0] and m[1] == n[0][-1] and m[2] == n[1][0] and m[3] == n[1][-1]:
                    self.correct_ds_h_num += 1

        for m in gold_ds_t:
            for n in pred_ds_t:
                if m[0] == n[0][0] and m[1] == n[0][-1] and m[2] == n[1][0] and m[3] == n[1][-1]:
                    self.correct_ds_t_num += 1

    def report_pair(self):
        p_holders = self.correct_ds_h_num / (self.pred_ds_h_num + 1e-18)
        r_holders = self.correct_ds_h_num / (self.gold_ds_h_num + 1e-18)
        f_holders = 2 * p_holders * r_holders / (p_holders + r_holders + 1e-18)

        p_targets = self.correct_ds_t_num / (self.pred_ds_t_num + 1e-18)
        r_targets = self.correct_ds_t_num / (self.gold_ds_t_num + 1e-18)
        f_targets = 2 * p_targets * r_targets / (p_targets + r_targets + 1e-18)

        return (0., 0., 0.), (p_holders, r_holders, f_holders), (p_targets, r_targets, f_targets)