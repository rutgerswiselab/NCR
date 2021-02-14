# coding=utf-8
import pandas as pd
from collections import Counter, defaultdict
import os
import numpy as np
import socket
from shutil import copyfile
from utils import global_p

np.random.seed(2018)


def random_split_data(all_data_file, dataset_name, vt_ratio=0.1, u_f=None, i_f=None):
    """
    随机切分已经生成的数据集文件 *.all.csv -> *.train.csv,*.validation.csv,*.test.csv
    :param all_data_file: 数据预处理完的文件 *.all.csv
    :param dataset_name: 给数据集起个名字
    :param vt_ratio: 验证集合测试集比例
    :param u_f: 用户特征文件 *.user.csv
    :param i_f: 物品特征文件 *.item.csv
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(global_p.DATASET_DIR, dataset_name)
    print('random_split_data', dir_name)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    vt_size = int(len(all_data) * vt_ratio)
    validation_set = all_data.sample(n=vt_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=vt_size).sort_index()
    train_set = all_data.drop(test_set.index)
    # print(train_set)
    # print(validation_set)
    # print(test_set)
    train_set.to_csv(os.path.join(dir_name, dataset_name + '.train.csv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, dataset_name + '.validation.csv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, dataset_name + '.test.csv'), index=False, sep='\t')

    # 复制用户、物品的特征文件
    if u_f is not None:
        copyfile(u_f, os.path.join(dir_name, dataset_name + '.user.csv'))
    if i_f is not None:
        copyfile(i_f, os.path.join(dir_name, dataset_name + '.item.csv'))
    return train_set, validation_set, test_set


def leave_out_by_time(all_data_file, dataset_name, leave_n=1, warm_n=5, u_f=None, i_f=None):
    """
    默认all_data里的交互是按时间顺序排列的，按交互顺序，把最后的交互划分到验证集合测试集里
    :param all_data_file: 数据预处理完的文件 *.all.csv，交互按时间顺序排列
    :param dataset_name: 给数据集起个名字
    :param leave_n: 验证和测试集保留几个用户交互
    :param warm_n: 保证测试用户在训练集中至少有warm_n个交互，否则交互全部放在训练集中
    :param u_f: 用户特征文件 *.user.csv
    :param i_f: 物品特征文件 *.item.csv
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(global_p.DATASET_DIR, dataset_name)
    print('leave_out_by_time', dir_name, leave_n, warm_n)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    min_label = all_data['label'].min()

    if min_label > 0:  # 所有交互皆为正向交互
        # 训练集保留每个用户的至少warm_n个正向交互，不够则全部保留，以保证验证集、测试集非冷启动
        train_set = all_data.groupby('uid').head(warm_n)
        all_data = all_data.drop(train_set.index)

        # 先满足test_set取倒数leave_n个
        test_set = all_data.groupby('uid').tail(leave_n)
        all_data = all_data.drop(test_set.index)
        # 再满足validation_set取剩下的倒数leave_n个
        validation_set = all_data.groupby('uid').tail(leave_n)
        all_data = all_data.drop(validation_set.index)
    else:  # 数据集中存在负向交互
        # 训练集保留每个用户的至少warm_n个正向交互，不够则全部保留，以保证验证集、测试集非冷启动
        train_set = []
        for uid, group in all_data.groupby('uid'):
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, 'label'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= warm_n:
                        break
            if found_idx > 0:
                train_set.append(group.loc[:found_idx + 1])
        train_set = pd.concat(train_set)
        all_data = all_data.drop(train_set.index)

        test_set = []
        for uid, group in all_data.groupby('uid'):  # 按uid聚合
            found, found_idx = 0, -1
            for idx in reversed(group.index):  # 倒着看该用户的历史，直到找到一个正样本
                if group.loc[idx, 'label'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:  # 如果找到了正样本，则该样本及其之后的负样本全部放入测试集
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        all_data = all_data.drop(test_set.index)

        validation_set = []
        for uid, group in all_data.groupby('uid'):  # 按uid聚合
            found, found_idx = 0, -1
            for idx in reversed(group.index):  # 倒着看该用户的历史，直到找到一个正样本
                if group.loc[idx, 'label'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:  # 如果找到了正样本，则该样本及其之后的负样本全部放入验证集
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        all_data = all_data.drop(validation_set.index)

    # 剩下的样本再放入训练集
    train_set = pd.concat([train_set, all_data]).sort_index()
    validation_set, test_set = validation_set.sort_index(), test_set.sort_index()

    train_set.to_csv(os.path.join(dir_name, dataset_name + '.train.csv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, dataset_name + '.validation.csv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, dataset_name + '.test.csv'), index=False, sep='\t')
    # 复制用户、物品的特征文件
    if u_f is not None:
        copyfile(u_f, os.path.join(dir_name, dataset_name + '.user.csv'))
    if i_f is not None:
        copyfile(i_f, os.path.join(dir_name, dataset_name + '.item.csv'))
    return train_set, validation_set, test_set
