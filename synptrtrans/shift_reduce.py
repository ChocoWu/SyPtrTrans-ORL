import numpy as np
import dynet as dy
import nn
import ops
from dy_utils import ParamManager as pm
from actions import Actions
from vocab import Vocab
import io_utils


class RoleLabeler(object):

    def __init__(self, config, encoder_output_dim, action_dict, role_type_dict):
        self.config = config
        self.model = pm.global_collection()
        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim

        # hidden_input_dim = lmda_dim * 5 + bi_rnn_dim * 2 + config['out_rnn_dim']
        hidden_input_dim = lmda_dim * 3 + bi_rnn_dim * 1 + config['out_rnn_dim']
        self.hidden_arg = nn.Linear(hidden_input_dim, config['output_hidden_dim'], activation='tanh')

        self.output_arg = nn.Linear(config['output_hidden_dim'], len(role_type_dict))

        hidden_input_dim_co = lmda_dim * 3 + bi_rnn_dim * 2 + config['out_rnn_dim']
        self.hidden_ent_corel = nn.Linear(hidden_input_dim_co, config['output_hidden_dim'],
                                          activation='tanh')
        self.output_ent_corel = nn.Linear(config['output_hidden_dim'], 2)

        self.position_embed = nn.Embedding(500, 20)

        attn_input = self.bi_rnn_dim * 1 + 20 * 2
        self.attn_hidden = nn.Linear(attn_input, 80, activation='tanh')
        self.attn_out = nn.Linear(80, 1)

        self.distrib_attn_hidden = nn.Linear(hidden_input_dim + len(role_type_dict), 80, activation='tanh')
        self.distrib_attn_out = nn.Linear(80, 1)
        self.empty_embedding = self.model.add_parameters((len(role_type_dict),), name='stackGuardEmb')

    def arg_prd_distributions_role_attn(self, inputs, arg_prd_distributions_role):
        inputs_ = [inputs for _ in range(len(arg_prd_distributions_role))]
        arg_prd_distributions_role = ops.cat(arg_prd_distributions_role, 1)
        inputs_ = ops.cat(inputs_, 1)
        att_input = dy.concatenate([arg_prd_distributions_role, inputs_], 0)
        hidden = self.distrib_attn_hidden(att_input)
        attn_out = self.distrib_attn_out(hidden)
        attn_prob = nn.softmax(attn_out, dim=1)
        rep = arg_prd_distributions_role * dy.transpose(attn_prob)
        return rep

    def forward(self, beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed, gold_role_label=None, role_table=None):

        state_embed = ops.cat([beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed], dim=0)

        hidden = self.hidden_arg(state_embed)
        out = self.output_arg(hidden)

        loss = dy.pickneglogsoftmax(out, gold_role_label)
        return loss

    def decode(self, beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed, role_table=None):

        state_embed = ops.cat([beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed], dim=0)

        hidden = self.hidden_arg(state_embed)
        out = self.output_arg(hidden)
        np_score = out.npvalue().flatten()
        return np.argmax(np_score)

    def position_aware_attn(self, hidden_mat, last_h, start1, ent1, start2, end2, seq_len):
        tri_pos_list = []
        ent_pos_list = []

        for i in range(seq_len):
            tri_pos_list.append(io_utils.relative_position(start1, ent1, i))
            ent_pos_list.append(io_utils.relative_position(start2, end2, i))

        tri_pos_emb = self.position_embed(tri_pos_list)
        tri_pos_mat = ops.cat(tri_pos_emb, 1)
        ent_pos_emb = self.position_embed(ent_pos_list)
        ent_pos_mat = ops.cat(ent_pos_emb, 1)

        att_input = ops.cat([hidden_mat, tri_pos_mat, ent_pos_mat], 0)

        hidden = self.attn_hidden(att_input)
        attn_out = self.attn_out(hidden)

        attn_prob = nn.softmax(attn_out, dim=1)

        rep = hidden_mat * dy.transpose(attn_prob)

        return rep


class ShiftReduce(object):

    def __init__(self, config, encoder_output_dim, action_dict, opi_role_vocab, dep_type_vocab=None,
                 udog_type_vocab=None, pos_type_vocab=None):

        self.config = config
        self.model = pm.global_collection()

        self.role_labeler = RoleLabeler(config, encoder_output_dim, action_dict, opi_role_vocab)
        self.role_null_id = opi_role_vocab[Vocab.NULL]
        self.role_type_dict = opi_role_vocab

        self.dep_type_vocab = dep_type_vocab
        self.udog_type_vocab = udog_type_vocab
        self.pos_type_vocab = pos_type_vocab
        self.pos_table = nn.Embedding(len(pos_type_vocab), config['pos_embed_dim'])

        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim

        dp_state = config['dp_state']
        dp_state_h = config['dp_state_h']

        # ------ states
        self.gamma = nn.LambdaVar(lmda_dim)  # the latest term
        self.sigma_e_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # opinion expression
        self.alpha_e_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # opinion expression
        self.sigma_o_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # opinion roles
        self.alpha_o_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # opinion roles
        self.actions_rnn = nn.StackLSTM(config['action_embed_dim'], config['action_rnn_dim'], dp_state, dp_state_h)
        self.out_rnn = nn.StackLSTM(bi_rnn_dim, config['out_rnn_dim'], dp_state, dp_state_h)  # from buffer
        # ------ states

        self.action_dict = action_dict
        self.act_table = nn.Embedding(len(action_dict), config['action_embed_dim'])
        self.role_table = nn.Embedding(len(opi_role_vocab), config['role_embed_dim'])  # holder & target type embedding
        self.ds_exp = nn.Embedding(2, config['ds_position_dim'])  # position 0/1
        self.length_table = nn.Embedding(50, config['length_embed_dim'])

        self.multi_gat = nn.MultiRCGA(config['n_rcga_layer'], config['rcga_in_features'],
                                      config['rcga_out_features'], config['rcga_dropout'],
                                      config['rcga_concat'])

        self.act = Actions(action_dict, opi_role_vocab)

        hidden_input_dim = bi_rnn_dim * 1 + lmda_dim * 4 \
                           + config['action_rnn_dim'] + config['out_rnn_dim']

        self.hidden_linear = nn.Linear(hidden_input_dim, config['output_hidden_dim'], activation='tanh')
        self.output_linear = nn.ActionGenerator(config['output_hidden_dim'],  len(action_dict))

        length_embed_dim = config['length_embed_dim']

        term_to_lmda_dim = bi_rnn_dim * 2 + length_embed_dim  # + config['sent_vec_dim']
        self.term_to_lmda = nn.Linear(term_to_lmda_dim, lmda_dim, activation='tanh')

        self.ds_ht_distributions_act = nn.Linear(lmda_dim + config['role_embed_dim'], len(action_dict),
                                                 activation='softmax')

        # beta
        self.empty_buffer_emb = self.model.add_parameters((bi_rnn_dim,), name='bufferGuardEmb')

        # pointer network parameter for ds
        self.pn2_linear = nn.Linear(config['pos_embed_dim']*2, 1)
        self.pn3_linear = nn.Linear(lmda_dim + bi_rnn_dim + config['pos_embed_dim'],
                                    1, activation='elu')

    def pointer_network(self, current_state, input_hidden_states, max_sent_len, idx, pos_list=None):
        pos_emb = self.pos_table(pos_list)
        input_exp = [ops.cat([x, y], dim=0) for x, y in zip(input_hidden_states, pos_emb)]
        x = [self.pn3_linear(ops.cat([t, current_state], dim=0)) for t in input_exp]

        pos_list.insert(0, 0)
        pos_list.append(0)
        z = [self.pn2_linear(ops.cat([self.pos_table[pos_list[idx]]-self.pos_table[pos_list[idx-1]], self.pos_table[pos_list[idx+1]]-self.pos_table[pos_list[idx]]], dim=0)) for idx in range(1, len(pos_list)-1)]
        att_score = [m + n for m, n in zip(z, x)]

        att_score = ops.cat(att_score, 0)
        valid_index = [x for x in range(idx, max_sent_len)]
        pn_log_probs = dy.log_softmax(att_score, valid_index)
        return pn_log_probs

    def __call__(self, toks, hidden_state_list, oracle_actions=None,
                 oracle_action_strs=None, is_train=True, sent_len=None, gold_dss=None, relations=None,
                 neibour_index=None, neibour_type=None, pos_list=None):

        def get_role_label(term_1_start_end, term_2_start_end):
            for i in relations:
                if term_1_start_end == i[:2] and term_2_start_end == i[2:4]:
                    return self.role_type_dict[i[4]]
            return self.role_null_id

        def get_dss_end_idx(start_idx):
            for d in gold_dss:
                if d[0] == start_idx:
                    return d[-1]
            return start_idx

        def get_ht_end_idx(start_idx):
            for i in relations:
                if i[2] == start_idx:
                    return i[3]
            return start_idx

        pred_dss = []
        holders = []
        targets = []
        pred_relations = []

        # beta, queue, for candidate sentence.
        graph = nn.Graph(self.config, neibour_index=neibour_index, neibour_type=neibour_type,
                         hidden_state_list=hidden_state_list,
                         sent_len=sent_len, udog_type_vocab=self.udog_type_vocab,
                         update_func=self.multi_gat)
        graph.update_hidden_state()
        buffer = nn.Buffer(self.bi_rnn_dim, hidden_state_list)
        losses = []
        loss_roles = []
        loss_position = []
        pred_action_strs = []

        ds_ht_distributions_ds = []

        self.actions_rnn.init_sequence(not is_train)
        self.out_rnn.init_sequence(not is_train)
        self.sigma_o_rnn.init_sequence(not is_train)
        self.sigma_e_rnn.init_sequence(not is_train)
        self.alpha_e_rnn.init_sequence(not is_train)
        self.alpha_o_rnn.init_sequence(not is_train)

        steps = 0
        while True:
            if steps >= len(oracle_actions):
                break
            if buffer.idx >= sent_len:
                break
            if buffer.is_empty() and self.gamma.is_empty():
                break

            # 上一个action
            pre_action = None if self.actions_rnn.is_empty() else self.actions_rnn.last_idx()

            # based on parser state, get valid actions.
            # only a very small subset of actions are valid, as below.
            valid_actions = []
            if not self.gamma.is_empty():
                valid_actions += [self.act.shift_id, self.act.arc_id, self.act.no_arc_id]
            elif pre_action is not None and self.act.is_no_start(pre_action):
                valid_actions += [self.act.no_start_id, self.act.ds_start_id, self.act.ht_start_id]
            elif pre_action is not None and self.act.is_ds_start(pre_action):
                valid_actions += [self.act.shift_id, self.act.arc_id, self.act.no_arc_id]
            elif pre_action is not None and self.act.is_ht_start(pre_action):
                valid_actions += [self.act.shift_id, self.act.arc_id, self.act.no_arc_id]
            elif pre_action is not None and self.act.is_no_arc(pre_action):
                valid_actions += [self.act.shift_id, self.act.arc_id, self.act.no_arc_id]
            elif pre_action is not None and self.act.is_arc(pre_action):
                valid_actions += [self.act.shift_id, self.act.arc_id, self.act.no_arc_id]
            elif pre_action is not None and self.act.is_shift(pre_action):
                valid_actions += [self.act.no_start_id, self.act.ds_start_id, self.act.ht_start_id]
            else:
                valid_actions += [self.act.no_start_id, self.act.ds_start_id, self.act.ht_start_id]

            # predicting action
            beta_embed = self.empty_buffer_emb if buffer.is_empty() else buffer.hidden_embedding()
            lmda_embed = self.gamma.embedding()
            sigma_e_embed = self.sigma_e_rnn.embedding()
            sigma_o_embed = self.sigma_o_rnn.embedding()
            alpha_e_embed = self.alpha_e_rnn.embedding()
            alpha_o_embed = self.alpha_o_rnn.embedding()
            action_embed = self.actions_rnn.embedding()
            out_embed = self.out_rnn.embedding()
            overall_rep = graph.graph_pool()

            state_embed = ops.cat([beta_embed, lmda_embed, sigma_e_embed,
                                   alpha_e_embed, action_embed, out_embed, overall_rep], dim=0)
            if is_train:
                state_embed = dy.dropout(state_embed, self.config['dp_out'])

            hidden_rep = self.hidden_linear(state_embed)

            logits = self.output_linear(hidden_rep, ds_ht_distributions_ds)
            log_probs = dy.log_softmax(logits, valid_actions)

            if is_train:
                try:
                    action = oracle_actions[steps]
                except IndexError as ie:
                    raise IndexError('steps: {}, oracle_actions length: {}, sent_len: {}'.format(steps, len(oracle_actions), sent_len))
                action_str = oracle_action_strs[steps]
                if action not in valid_actions:
                    raise RuntimeError('Action %s dose not in valid_actions, %s(pre) %s: [%s]' % (
                        action_str, self.act.to_act_str(pre_action),
                        self.act.to_act_str(action), ','.join([self.act.to_act_str(ac) for ac in valid_actions])))
                losses.append(dy.pick(log_probs, action))
            else:
                np_log_probs = log_probs.npvalue()
                action = np.argmax(np_log_probs)
                action_str = self.act.to_act_str(action)
                pred_action_strs.append(action_str)

            # if True:continue
            # update the parser state according to the action.
            if self.act.is_no_start(action):
                if not buffer.is_empty():
                    hx, idx = buffer.pop()
                    self.out_rnn.push(hx, idx, 'None')
            elif self.act.is_ds_start(action):
                if not buffer.is_empty():
                    ds_start_hx, ds_start_idx = buffer.last_state()
                    self.out_rnn.push(ds_start_hx, ds_start_idx, 'None')
                    scores = self.pointer_network(ds_start_hx, hidden_state_list, sent_len, ds_start_idx, pos_list=pos_list)
                    if is_train:
                        ds_end_idx = get_dss_end_idx(ds_start_idx)
                        loss_position.append(dy.pick(scores, ds_end_idx))
                    else:
                        np_scores = scores.npvalue()
                        ds_end_idx = int(np.argmax(np_scores))
                    if ds_end_idx <= ds_start_idx:
                        ds_end_idx = ds_start_idx
                    length_emb = self.length_table(ds_end_idx - ds_start_idx + 1)
                    ds_rep = self.term_to_lmda(ops.cat([buffer.hidden_states[ds_start_idx], buffer.hidden_states[ds_end_idx], length_emb], dim=0))
                    self.gamma.push(ds_rep, [ds_start_idx, ds_end_idx], self.gamma.OPINION)
                    pred_dss.append([ds_start_idx, ds_end_idx])
            elif self.act.is_ht_start(action):
                if not buffer.is_empty():
                    ht_start_hx, ht_start_idx = buffer.last_state()
                    self.out_rnn.push(ht_start_hx, ht_start_idx, 'None')

                    scores = self.pointer_network(ht_start_hx, hidden_state_list, sent_len, ht_start_idx, pos_list=pos_list)
                    if is_train:
                        ht_end_idx = get_ht_end_idx(ht_start_idx)
                        loss_position.append(dy.pick(scores, ht_end_idx))
                    else:
                        np_scores = scores.npvalue()
                        ht_end_idx = int(np.argmax(np_scores))
                    if ht_end_idx <= ht_start_idx:
                        ht_end_idx = ht_start_idx
                    length_emb = self.length_table(ht_end_idx - ht_start_idx + 1)
                    ht_rep = self.term_to_lmda(ops.cat([buffer.hidden_states[ht_start_idx], buffer.hidden_states[ht_end_idx], length_emb], dim=0))
                    self.gamma.push(ht_rep, [ht_start_idx, ht_end_idx], self.gamma.HT)
            elif self.act.is_arc(action):
                lmda_idx = self.gamma.idx
                lmda_embed = self.gamma.embedding()
                lmda_type = self.gamma.lambda_type
                if lmda_type == 'opinion':
                    if not self.sigma_o_rnn.is_empty():
                        sigma_last_embed, sigma_last_idx, sigma_last_type = self.sigma_o_rnn.pop()
                        if is_train:
                            role_label = get_role_label(lmda_idx, sigma_last_idx)
                            loss_role = self.role_labeler.forward(beta_embed, lmda_embed, sigma_last_embed,
                                                                  alpha_o_embed, out_embed, role_label,
                                                                  role_table=self.role_table)
                            loss_roles.append(loss_role)
                        else:
                            role_label = self.role_labeler.decode(beta_embed, lmda_embed, sigma_last_embed,
                                                                  alpha_e_embed,
                                                                  out_embed, role_table=self.role_table)
                            ht_role = self.role_type_dict.get_token(role_label)
                            if ht_role == 'HOLDER':
                                if sigma_last_idx not in holders:
                                    holders.append(sigma_last_idx)
                                frame = (lmda_idx, sigma_last_idx, ht_role)
                                pred_relations.append(frame)
                                # update UDOG
                                graph.update_opi(lmda_idx[0], lmda_idx[-1], sigma_last_idx[0], sigma_last_idx[-1],
                                                 ht_role)
                            elif ht_role == 'TARGET':
                                if sigma_last_idx not in targets:
                                    targets.append(sigma_last_idx)
                                frame = (lmda_idx, sigma_last_idx, ht_role)
                                pred_relations.append(frame)
                                # update UDOG
                                graph.update_opi(lmda_idx[0], lmda_idx[-1], sigma_last_idx[0], sigma_last_idx[-1],
                                                 ht_role)
                            else:
                                pass
                        # update representation via rcga
                        graph.update_hidden_state()
                        self.alpha_o_rnn.push(sigma_last_embed, sigma_last_idx, self.role_type_dict.get_token(role_label))
                elif lmda_type == 'ht':
                    if not self.sigma_e_rnn.is_empty():
                        sigma_last_embed, sigma_last_idx, sigma_last_type = self.sigma_e_rnn.pop()
                        if is_train:
                            role_label = get_role_label(sigma_last_idx, lmda_idx)
                            loss_role = self.role_labeler.forward(beta_embed, lmda_embed, sigma_last_embed,
                                                                  alpha_e_embed, out_embed, role_label,
                                                                  role_table=self.role_table)
                            loss_roles.append(loss_role)
                        else:
                            role_label = self.role_labeler.decode(beta_embed, lmda_embed, sigma_last_embed,
                                                                  alpha_e_embed,
                                                                  out_embed, role_table=self.role_table)
                            ht_role = self.role_type_dict.get_token(role_label)
                            if ht_role == 'HOLDER':
                                if lmda_idx not in holders:
                                    holders.append(lmda_idx)
                                frame = (sigma_last_idx, lmda_idx, ht_role)
                                pred_relations.append(frame)
                                graph.update_opi(sigma_last_idx[0], sigma_last_idx[-1], lmda_idx[0], lmda_idx[-1], ht_role)
                            elif ht_role == 'TARGET':
                                if lmda_idx not in targets:
                                    targets.append(lmda_idx)
                                frame = (sigma_last_idx, lmda_idx, ht_role)
                                pred_relations.append(frame)
                                graph.update_opi(sigma_last_idx[0], sigma_last_idx[-1], lmda_idx[0], lmda_idx[-1], ht_role)
                            else:
                                pass
                        self.alpha_e_rnn.push(sigma_last_embed, sigma_last_idx, sigma_last_type)
                        # update representation via rcga
                        graph.update_hidden_state()
                else:
                    raise RuntimeError('Wrong lambda type, not opinion or expression')

            elif self.act.is_no_arc(action):
                lmda_type = self.gamma.lambda_type
                if lmda_type == 'opinion':
                    if not self.sigma_o_rnn.is_empty():
                        hx, idx, tx = self.sigma_o_rnn.pop()
                        self.alpha_o_rnn.push(hx, idx, tx)
                elif lmda_type == 'ht':
                    if not self.sigma_e_rnn.is_empty():
                        hx, idx, tx = self.sigma_e_rnn.pop()
                        self.alpha_e_rnn.push(hx, idx, tx)
                else:
                    raise RuntimeError('Wrong lambda type, not aspect or opinion')

            elif self.act.is_shift(action):
                # while no elements are in sigma
                while not self.alpha_e_rnn.is_empty():
                    self.sigma_e_rnn.push(*self.alpha_e_rnn.pop())
                while not self.alpha_o_rnn.is_empty():
                    self.sigma_o_rnn.push(*self.alpha_o_rnn.pop())
                while not self.gamma.is_empty():
                    if self.gamma.lambda_type == 'opinion':
                        self.sigma_e_rnn.push(*self.gamma.pop())
                    elif self.gamma.lambda_type == 'ht':
                        self.sigma_o_rnn.push(*self.gamma.pop())
                    else:
                        raise RuntimeError('Wrong lambda type, not aspect or opinion')
                if not buffer.is_empty():
                    _ = buffer.pop()
            else:
                raise RuntimeError('Unknown action type:' + str(action) + self.act.to_act_str(action))

            steps += 1
            self.actions_rnn.push(self.act_table[action], action, 'None')

        self.clear()

        return losses, loss_roles, loss_position, pred_dss, holders, targets, pred_relations, pred_action_strs

    def clear(self):
        self.sigma_e_rnn.clear()
        self.sigma_o_rnn.clear()
        self.alpha_o_rnn.clear()
        self.alpha_e_rnn.clear()
        self.actions_rnn.clear()
        self.gamma.clear()
        self.out_rnn.clear()
