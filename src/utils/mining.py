# coding=utf-8

import pandas as pd


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
