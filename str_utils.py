

import re


def normalize_spectial_symbol(sent):
    sent = str(sent).replace('``', '"')
    sent = str(sent).replace("''", '"')
    sent = str(sent).replace('-LRB-', '(')
    sent = str(sent).replace('-RRB-', ')')
    sent = str(sent).replace('-LSB-', '(')
    sent = str(sent).replace('-RSB-', ')')
    return sent


def collapse_role_type(role_type):
    '''
    collapse role types from 36 to 28 following Bishan Yang 2016
    we also have to handle types like 'Beneficiary#Recipient'
    :param role_type:
    :return:
    '''
    if role_type.startswith('Time-'):
        return 'Time'
    idx = role_type.find('#')
    if idx != -1:
        role_type = role_type[:idx]

    return role_type


# def normalize_tok(tok, lower=False, normalize_digits=False):
#
#     if lower:
#         tok = tok.lower()
#     if normalize_digits:
#         tok = re.sub(r"\d", "0", tok)
#         tok = re.sub(r"^(\d+[,])*(\d+)$", "0", tok)
#     return tok

def normalize_tok(tok, lower=True, normalize_digits=True):

    if lower:
        tok = normalize_spectial_symbol(tok).lower().strip()
    if normalize_digits:
        RE_NUM = r"\b\d+(?:[\.,']\d+)?\b"
        RE_PERCENTAGE = RE_NUM + "%"
        tok = re.sub(r"\d", "0", tok)
        tok = re.sub(r"^(\d+[,])*(\d+)$", "0", tok)
        tok = re.sub(RE_NUM, '0', tok)
        tok = re.sub(RE_PERCENTAGE, '0', tok)
        tok = re.sub(r"%", 'percentage', tok)
        tok = re.sub(r"e\.g\.", 'for example', tok)

    return tok


def reshape_dependency_tree(ori_token,  pos_list, dep_list, dep_head, ds):
    """
    reshape the ds-oriented dependency tree, and then store the data according to the result of pre-order traversal.
    :param ori_token: list
    :param pos_list: list
    :param dep_list: list
    :param dep_head: list
    :param ds: list token not only including the [ds-start, ds-end]
    :return:
    """
    token_len = len(ori_token)
    new_token, new_pos_list, new_dep_list, new_dep_head = [], [], [], []
    for i in ds:
        new_token.append(ori_token[i])
        new_pos_list.append(pos_list[i])
        new_dep_list.append(dep_list[i])
        new_dep_head.append(dep_head[i])
        # del ori_token[i], pos_list[i], dep_list[i], dep_head[i]
    while len(new_token) < token_len:
        # for i, j, m, n in zip(ori_token, pos_list, dep_list, dep_head):
        k = len(ori_token)
        # del_index = []
        for i in range(k):
            if dep_head[i] in new_dep_head and ori_token[i] not in new_token:
                new_token.append(ori_token[i])
                new_pos_list.append(pos_list[i])
                new_dep_list.append(dep_list[i])
                new_dep_head.append(dep_head[i])
                # del_index.append(i)
        # for i in del_index:
        #     del ori_token[i], pos_list[i], dep_list[i], dep_head[i]
    return new_token, new_pos_list, new_dep_list, new_dep_head


def capitalize_first_char(sent):
    sent = str(sent[0]).upper() + sent[1:]
    return sent





