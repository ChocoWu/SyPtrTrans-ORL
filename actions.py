# -*- coding: utf-8 -*-


class Actions(object):

    no_start = 'NO-START'
    ds_start = 'DS-START'
    ht_start = 'HT_START'
    shift = 'SHIFT'
    arc = 'ARC'
    no_arc = 'NO_ARC'

    def __init__(self, action_dict, role_type_dict, with_copy_shift=True):
        self.no_start_id = action_dict[Actions.no_start]
        self.ds_start_id = action_dict[Actions.ds_start]
        self.ht_start_id = action_dict[Actions.ht_start]
        self.arc_id = action_dict[Actions.arc]
        self.no_arc_id = action_dict[Actions.no_arc]
        self.shift_id = action_dict[Actions.shift]

        self.act_id_to_str = {v: k for k, v in action_dict.items()}
        self.act_str_to_id = action_dict

    def to_act_str(self, act_id):
        return self.act_id_to_str[act_id]

    def to_act_id(self, act_str):
        return self.act_str_to_id[act_str]

    def is_ds_start(self, act_id):
        return self.ds_start_id == act_id

    def is_no_start(self, act_id):
        return self.no_start_id == act_id

    def is_ht_start(self, act_id):
        return self.ht_start_id == act_id

    def is_shift(self, act_id):
        return self.shift_id == act_id

    def is_arc(self, act_id):
        return self.arc_id == act_id

    def is_no_arc(self, act_id):
        return self.no_arc_id == act_id

    @staticmethod
    def relation_exists(relations, r1, r2):
        for r in relations:
            if (r1 == r[:2] and r2 == r[2:4]) or (r2 == r[:2] and r1 == r[2:4]):
                return True
        return False

    @staticmethod
    def make_oracle(tokens, sorted_term_dic, relations):

        actions = []

        sent_length = len(tokens)
        start_idxs = [x[0] for x in sorted_term_dic.values()]
        dss = []
        ht = []
        i = 0
        flag = 0
        while i < sent_length:
            for k, v in sorted_term_dic.items():
                if v[0] == i:
                    flag = 1
                    if k.startswith('D'):
                        actions.append(Actions.ds_start)
                        for m in reversed(ht):
                            if Actions.relation_exists(relations, v, m):
                                actions.append(Actions.arc)
                            else:
                                actions.append(Actions.no_arc)
                        actions.append(Actions.shift)
                        dss.append(v)
                    else:
                        actions.append(Actions.ht_start)
                        for m in reversed(dss):
                            if Actions.relation_exists(relations, m, v):
                                actions.append(Actions.arc)
                            else:
                                actions.append(Actions.no_arc)
                        actions.append(Actions.shift)
                        ht.append(v)
                    break
            if flag == 1:
                flag = 0
                pass
            else:
                actions.append(Actions.no_start)
            i += 1

        return actions


if __name__ == '__main__':
    actions = Actions.make_oracle([x for x in range(100)], [[28, 29], [65, 66, 67], [8], [47], [91]], [[27], [64], [7]],
                                  [[22, 23], [59, 60, 61, 62], [5], [48]])
    print(len(actions))




