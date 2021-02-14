# coding=utf-8

import pandas as pd

from pymining import itemmining, assocrules


def group_user_interactions_csv(in_csv, out_csv, label='label', sep='\t'):
    print('group_user_interactions_csv', out_csv)
    all_data = pd.read_csv(in_csv, sep=sep)
    group_inters = group_user_interactions_df(in_df=all_data, label=label)
    group_inters.to_csv(out_csv, sep=sep, index=False)
    return group_inters


def group_user_interactions_df(in_df, label='label', seq_sep=','):
    all_data = in_df
    if label in all_data.columns:
        all_data = all_data[all_data[label] > 0]
    uids, inters = [], []
    for name, group in all_data.groupby('uid'):
        uids.append(name)
        inters.append(seq_sep.join(group['iid'].astype(str).tolist()))
    group_inters = pd.DataFrame()
    group_inters['uid'] = uids
    group_inters['iids'] = inters
    return group_inters


def mine_item_rules(in_csv, out_csv, min_support=5, min_confidence=0.5):
    print('mine_item_rules', out_csv)
    his_df = pd.read_csv(in_csv, sep='\t')

    # transactions = perftesting.get_default_transactions()
    # print(transactions)
    transactions = his_df['iids'].tolist()
    transactions = [t.split(',') for t in transactions]

    relim_input = itemmining.get_relim_input(transactions)
    print('build relim_input done')
    item_sets = itemmining.relim(relim_input, min_support=min_support)
    print('relim done')

    # fptree = itemmining.get_fptree(transactions=transactions, min_support=min_support)
    # print('build fptree done')
    # item_sets = itemmining.fpgrowth(fptree, min_support=min_support)
    # print('fpgrowth done')

    # print(item_sets)
    rules = assocrules.mine_assoc_rules(item_sets, min_support=min_support, min_confidence=min_confidence)
    lefts, rights, supports, confidences = [], [], [], []
    left_set, right_set = set(), set()
    for rule in rules:
        left, right, support, confidence = rule
        lefts.append(','.join(list(left)))
        rights.append(','.join(list(right)))
        supports.append(support)
        confidences.append(confidence)
        left_set = left_set.union(left)
        right_set = right_set.union(right)
    # print(left_set)
    # print(right_set)
    print('left unique: %d, right unique: %d, total unique: %d'
          % (len(left_set), len(right_set), len(left_set.union(right_set))))
    out_df = pd.DataFrame()
    out_df['left'] = lefts
    out_df['right'] = rights
    out_df['support'] = supports
    out_df['confidence'] = confidences
    out_df = out_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
    print('# rules: %d' % len(out_df))
    out_df.to_csv(out_csv, sep='\t', index=False)
    return out_df
