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
    Split train/validation/test by timestamp.
    By default, the interactions in all_data_file are already sorted by timestamp.
    :param all_data_file: preprocessed dataset file *.all.csv，which is sorted by timestamp.
    :param dataset_name: dataset name (used as the processed dataset name)
    :param leave_n: number of items that are left in validation and test set.
    :param warm_n: minimum number of interactions to leave in training dataset for each user.
    :param u_f: user feature file (not used here)
    :param i_f: item feature file (not used here)
    :return: pandas dataframe for training/validation/test sets
    """
    dir_name = os.path.join(global_p.DATASET_DIR, dataset_name)
    print('leave_out_by_time', dir_name, leave_n, warm_n)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    min_label = all_data['label'].min()

    if min_label > 0:  # all the interactions are positive
        '''
        Keep at least 'warm_n' number of interactions in training dataset. 
        If user has less than 'warm_n' interactions, then keep all the interactions in training set.
        This is to guarantee that no cold start issue for validation and testing.
        '''
        train_set = all_data.groupby('uid').head(warm_n)
        all_data = all_data.drop(train_set.index)

        # Get last leave_n items to put into test_set
        test_set = all_data.groupby('uid').tail(leave_n)
        all_data = all_data.drop(test_set.index)
        # Get validation_set from the remaining last leave_n items
        validation_set = all_data.groupby('uid').tail(leave_n)
        all_data = all_data.drop(validation_set.index)
    else:  # there are negative interactions
        '''
        Keep at least 'warm_n' number of interactions in training dataset. 
        If user has less than 'warm_n' interactions, then keep all the interactions in training set.
        This is to guarantee that no cold start issue for validation and testing.
        '''
        train_set = []
        # generate training set by looking for the first warm_n POSITIVE interactions
        for uid, group in all_data.groupby('uid'):  # group by uid
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
        # drop the training data info
        all_data = all_data.drop(train_set.index)

        # generate test set by looking for the last leave_n POSITIVE interactions
        test_set = []
        for uid, group in all_data.groupby('uid'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'label'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        all_data = all_data.drop(test_set.index)

        validation_set = []
        for uid, group in all_data.groupby('uid'):  #
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'label'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            # put all the negative interactions encountered during the search process into validation set
            if found_idx > 0:
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        all_data = all_data.drop(validation_set.index)

    # The remaining data (after removing validation and test) are all in training data
    train_set = pd.concat([train_set, all_data]).sort_index()
    validation_set, test_set = validation_set.sort_index(), test_set.sort_index()

    train_set.to_csv(os.path.join(dir_name, dataset_name + '.train.csv'), index=False, sep='\t')
    validation_set.to_csv(os.path.join(dir_name, dataset_name + '.validation.csv'), index=False, sep='\t')
    test_set.to_csv(os.path.join(dir_name, dataset_name + '.test.csv'), index=False, sep='\t')
    # Copy feature info if applicable
    if u_f is not None:
        copyfile(u_f, os.path.join(dir_name, dataset_name + '.user.csv'))
    if i_f is not None:
        copyfile(i_f, os.path.join(dir_name, dataset_name + '.item.csv'))
    return train_set, validation_set, test_set
