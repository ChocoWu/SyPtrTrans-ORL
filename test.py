#!/usr/bin/env python
# encoding: utf-8

import json
import sys

from collections import OrderedDict, Counter
from nltk.parse import CoreNLPDependencyParser

depparser = CoreNLPDependencyParser(url='http://127.0.0.1:9000')


def get_dep(token_list):
    res = []
    parser_res = depparser.parse(token_list)
    # for i in parser_res:
    #     res.append(i.to_conll(4).split('\t'))
    for i in parser_res:
        temp = i.to_conll(4).strip().split('\n')
        for t in temp:
            res.append(t.split('\t'))
    return res


HOLDER = "HOLDER"
TARGET = "TARGET"


class Sample(object):
    def __init__(self, obj):
        self.sentence = obj["sentence"]
        self.gold_orl = obj['gold_orl']
        self.sys_orl = obj["sys_orl"]


def load_eval_data(eval_path):
    with open(eval_path, 'r') as f:
        eval_data = [Sample(json.loads(jsonline.split('\n')[0])) for jsonline in f.readlines()]

    print("Loaded {} eval examples.".format(len(eval_data)))
    return eval_data


class EvalMetric(object):
    def __init__(self, name="None"):
        self.name = name
        self.matched, self.sys, self.gold = 0, 0, 0
        self.p = self.r = self.f = 0.0

    def compute_prf(self):
        try:
            self.p = 100.0 * self.matched / self.sys
        except:
            self.p = 0.0
        try:
            self.r = 100.0 * self.matched / self.gold
        except:
            self.r = 0.0
        try:
            self.f = 2.0 * self.p * self.r / (self.p + self.r)
        except:
            self.f = 0.0
        print("="*5, self.name, "="*5)
        print("Precision:", self.matched, '/', self.sys, '=', self.p)
        print("Recall:", self.matched, '/', self.gold, '=', self.r)
        print("F1 score:", self.f)
        # print('%.5f\t%.5f\t%.5f' % (self.p, self.r, self.f))


def analyze_error_prediction_matrix(samples):
    agent_binary, target_binary, all_binary = EvalMetric("Agent"), EvalMetric("Target"), EvalMetric("All")
    agent_proportional, target_proportional, all_proportional = \
        EvalMetric("Agent"), EvalMetric("Target"), EvalMetric("All")
    agent_exact, target_exact, all_exact = EvalMetric("Agent"), EvalMetric("Target"), EvalMetric("All")
    exp_binary, exp_proportional, exp_exact = EvalMetric("Exp-binary"), EvalMetric("Exp-proportional"), \
                                              EvalMetric("Exp-exact")
    all34, all57, all812, all13 = EvalMetric("All"), EvalMetric("All"), EvalMetric("All"), EvalMetric("All")
    all1, all2 = EvalMetric("All"), EvalMetric("All")
    # all67, all810, all1113, all14 = EvalMetric("All"), EvalMetric("All"), EvalMetric("All"), EvalMetric("All")
    for i, sample in enumerate(samples):
        gold_orl, sys_orl = sample.gold_orl, sample.sys_orl
        # dict s-e: label
        dict_gold_orl, dict_sys_orl = OrderedDict(), OrderedDict()
        for g_orl in gold_orl:  # construct the expression-argument tuples
            dse_s, dse_e, s, e, label = g_orl
            expression = (dse_s, dse_e)
            # expression = str(dse_s) + '-' + str(dse_e)
            argument = (s, e, label)
            if expression not in dict_gold_orl:
                dict_gold_orl[expression] = []
                dict_gold_orl[expression].append(argument)
            else:
                dict_gold_orl[expression].append(argument)
        for s_orl in sys_orl:  # construct the expression-argument tuples
            dse_s, dse_e, s, e, label = s_orl
            expression = (dse_s, dse_e)
            # expression = str(dse_s) + '-' + str(dse_e)
            argument = (s, e, label)
            if expression not in dict_sys_orl:
                dict_sys_orl[expression] = []
                dict_sys_orl[expression].append(argument)
            else:
                dict_sys_orl[expression].append(argument)

        # expression evaluation
        for gold_expression in dict_gold_orl:
            exp_binary.gold += 1
            exp_proportional.gold += 1
            exp_exact.gold += 1
        for sys_expression in dict_sys_orl:
            exp_binary.sys += 1
            exp_proportional.sys += 1
            exp_exact.sys += 1
        for gold_expression in dict_gold_orl:  # Exact
            for sys_expression in dict_sys_orl:
                if gold_expression == sys_expression:
                    exp_exact.matched += 1
                    break
        for gold_expression in dict_gold_orl:  # binary
            g_s, g_e = gold_expression
            for sys_expression in dict_sys_orl:
                s_s, s_e = sys_expression
                find = False
                for interval in range(g_s, g_e + 1):
                    if s_s <= interval <= s_e:
                        exp_binary.matched += 1
                        find = True
                        break
                if find is True:
                    break

        for sys_expression in dict_sys_orl:
            s_s, s_e = sys_expression
            matched_proportions = []
            for gold_expression in dict_gold_orl:  # proportional
                g_s, g_e = gold_expression

                count = 0
                for interval in range(g_s, g_e + 1):
                    if s_s <= interval <= s_e:
                        count += 1
                if count > 0:
                    matched_proportions.append(1.0 * count / (g_e - g_s + 1))
            if len(matched_proportions):
                exp_proportional.matched += max(matched_proportions)
        """===split==="""

        for expression in dict_gold_orl:  # compute the gold
            for argument in dict_gold_orl[expression]:
                s, e, label = argument
                all_binary.gold += 1
                all_proportional.gold += 1
                all_exact.gold += 1
                if min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) == 1:
                    all1.gold += 1
                elif min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) == 2:
                    all2.gold += 1
                elif 3 <= min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) <= 4:
                    all34.gold += 1
                elif 5 <= min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) <= 7:
                    all57.gold += 1
                elif 8 <= min(abs(expression[0]-argument[1]), abs(expression[1]-argument[0])) < 12:
                    all812.gold += 1
                else:
                    all13.gold += 1
                if label == HOLDER:
                    agent_binary.gold += 1
                    agent_proportional.gold += 1
                    agent_exact.gold += 1
                else:
                    assert label == TARGET
                    target_binary.gold += 1
                    target_proportional.gold += 1
                    target_exact.gold += 1

        for expression in dict_sys_orl:  # compute the sys
            for argument in dict_sys_orl[expression]:
                s, e, label = argument
                all_binary.sys += 1
                all_proportional.sys += 1
                all_exact.sys += 1
                # if (argument[1]-argument[0]+1) < 5:
                #     all14.sys += 1
                # elif 4 < (argument[1]-argument[0]+1) < 9:
                #     all58.sys += 1
                # elif 8 < (argument[1]-argument[0]+1) < 16:
                #     all915.sys += 1
                # else:
                #     all16.sys += 1
                if min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) == 1:
                    all1.sys += 1
                elif min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) == 2:
                    all2.sys += 1
                elif 3 <= min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) <= 4:
                    all34.sys += 1
                elif 5 <= min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) <= 7:
                    all57.sys += 1
                elif 8 <= min(abs(expression[0] - argument[1]), abs(expression[1] - argument[0])) < 12:
                    all812.sys += 1
                else:
                    all13.sys += 1

                if label == HOLDER:
                    agent_binary.sys += 1
                    agent_proportional.sys += 1
                    agent_exact.sys += 1
                elif label == TARGET:
                    target_binary.sys += 1
                    target_proportional.sys += 1
                    target_exact.sys += 1
                else:
                    # assert label == dse
                    pass

        for expression in sorted(list(dict_sys_orl.keys())):  # compute the sys
            if expression not in dict_gold_orl:  # debug: some gold orl has no argument, only expression
                # print(sample.sentence)
                continue
            gold_arguments = dict_gold_orl[expression]
            for argument in dict_sys_orl[expression]:
                s, e, label = argument
                if argument in gold_arguments:  # exact
                    all_binary.matched += 1
                    # all_proportional.matched += 1
                    all_exact.matched += 1
                    # if (argument[1]-argument[0]+1) < 5:
                    #     all14.matched += 1
                    # elif 4 < (argument[1]-argument[0]+1) < 9:
                    #     all58.matched += 1
                    # elif 8 < (argument[1]-argument[0]+1) < 16:
                    #     all915.matched += 1
                    # else:
                    #     all16.matched += 1
                    if min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) == 1:
                        all1.matched += 1
                    elif min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) == 2:
                        all2.matched += 1
                    elif 3 <= min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) <= 4:
                        all34.matched += 1
                    elif 5 <= min(abs(expression[0] - argument[1]), abs(expression[0] - argument[1])) <= 7:
                        all57.matched += 1
                    elif 8 <= min(abs(expression[0] - argument[1]), abs(expression[1] - argument[0])) < 12:
                        all812.matched += 1
                    else:
                        all13.matched += 1
                    if label == HOLDER:
                        agent_binary.matched += 1
                        # agent_proportional.matched += 1
                        agent_exact.matched += 1
                    else:
                        assert label == TARGET
                        target_binary.matched += 1
                        # target_proportional.matched += 1
                        target_exact.matched += 1
                else:
                    # binary
                    find = False
                    for index in range(s, e + 1):
                        for gold_arg in gold_arguments:
                            g_s, g_e, g_label = gold_arg
                            if g_label == label:
                                if g_s <= index <= g_e:
                                    all_binary.matched += 1
                                    if label == HOLDER:
                                        agent_binary.matched += 1
                                    else:
                                        target_binary.matched += 1
                                    find = True
                                    break
                        if find is True:
                            break
                # proportional: for sys: compare gold, not for gold: compare sys! important TODO
                list_of_proportional = []
                for gold_argument in dict_gold_orl[expression]:
                    g_s, g_e, g_label = gold_argument
                    matched_positions = 0
                    if label != g_label:
                        pass
                    else:
                        for position in range(g_s, g_e + 1):
                            if s <= position <= e:
                                matched_positions += 1
                        list_of_proportional.append(1.0 * matched_positions / (g_e - g_s + 1))
                if len(list_of_proportional) > 0:  # matched a gold argument
                    all_proportional.matched += max(list_of_proportional)
                    if label == HOLDER:
                        agent_proportional.matched += max(list_of_proportional)
                    else:
                        target_proportional.matched += max(list_of_proportional)
                    # if max(list_of_proportional) > 0:
                    #     print(i, expression, max(list_of_proportional))

    print("="*15, 'Binary Metric', "="*15)
    agent_binary.compute_prf()
    target_binary.compute_prf()
    all_binary.compute_prf()

    print("="*15, 'Proportional Metric', "="*15)
    agent_proportional.compute_prf()
    target_proportional.compute_prf()
    all_proportional.compute_prf()

    print("="*15, 'Exact Metric', "="*15)
    agent_exact.compute_prf()
    target_exact.compute_prf()
    all_exact.compute_prf()

    print("="*15, 'expression Binary Metric', "="*15)
    exp_binary.compute_prf()
    print("="*15, 'expression Proportional Metric', "="*15)
    exp_proportional.compute_prf()
    print("="*15, 'expression Exact Metric', "="*15)
    exp_exact.compute_prf()

    return ((agent_binary, target_binary, all_binary),
            (agent_proportional, target_proportional, all_proportional),
            (agent_exact, target_exact, all_exact),
            (exp_binary, exp_proportional, exp_exact))


if __name__ == '__main__':

    for i in range(5):
        print("="*15, 'fold-{}'.format(i), "="*15)
        data = load_eval_data('data/eval/{}/results.txt'.format(i))
        analyze_error_prediction_matrix(data)
        print('\n')
