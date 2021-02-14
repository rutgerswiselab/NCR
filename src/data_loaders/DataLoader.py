# coding=utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import logging
from utils.mining import group_user_interactions_df
from utils import global_p
import json


class DataLoader(object):
    """
    Only load data file and information file.
    """

    @staticmethod
    def parse_data_args(parser):
        """
        data loader related command line arguments parser
        :param parser:
        :return:
        """
        parser.add_argument('--path', type=str, default='../dataset/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml100k-1-5',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--label', type=str, default='label',
                            help='name of dataset label column.')
        return parser

    def __init__(self, path, dataset, label='label', load_data=True, sep='\t', seqs_sep=','):
        """
        Initialization
        :param path: dataset path
        :param dataset: dataset name
        :param label: label column name
        :param load_data: True for load dataset file, False for only load info file
        :param sep: the separator for .csv file
        :param seqs_sep: the separator for negative/history sequence, for example: "1,2,3"
        """
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.train_file = os.path.join(self.path, dataset + global_p.TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset + global_p.VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset + global_p.TEST_SUFFIX)
        self.info_file = os.path.join(self.path, dataset + global_p.INFO_SUFFIX)
        self.user_file = os.path.join(self.path, dataset + global_p.USER_SUFFIX)
        self.item_file = os.path.join(self.path, dataset + global_p.ITEM_SUFFIX)
        self.train_his_file = os.path.join(self.path, dataset + global_p.TRAIN_GROUP_SUFFIX)
        self.vt_his_file = os.path.join(self.path, dataset + global_p.VT_GROUP_SUFFIX)
        self.sep, self.seqs_sep = sep, seqs_sep
        self.load_data = load_data
        self.label = label

        self.train_df, self.validation_df, self.test_df = None, None, None
        self._load_user_item()
        self._load_data()
        self._load_his()
        self._load_info()

    def _load_user_item(self):
        """
        load user .csv file
        :return:
        """
        self.user_df, self.item_df = None, None
        if os.path.exists(self.user_file) and self.load_data:
            logging.info("load user csv...")
            self.user_df = pd.read_csv(self.user_file, sep='\t')
        if os.path.exists(self.item_file) and self.load_data:
            logging.info("load item csv...")
            self.item_df = pd.read_csv(self.item_file, sep='\t')

    def _load_data(self):
        """
        load train, validation, test .csv files
        :return:
        """
        if os.path.exists(self.train_file) and self.load_data:
            logging.info("load train csv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            logging.info("size of train: %d" % len(self.train_df))
        if os.path.exists(self.validation_file) and self.load_data:
            logging.info("load validation csv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logging.info("size of validation: %d" % len(self.validation_df))
        if os.path.exists(self.test_file) and self.load_data:
            logging.info("load test csv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            logging.info("size of test: %d" % len(self.test_df))

    def _load_info(self):
        """
        load dataset information file. If not found then create one.
        :return:
        """

        def json_type(o):
            if isinstance(o, np.int64):
                return int(o)
            # if isinstance(o, np.float32): return int(o)
            raise TypeError

        max_dict, min_dict = {}, {}
        if not os.path.exists(self.info_file):
            for df in [self.train_df, self.validation_df, self.test_df, self.user_df, self.item_df]:
                if df is None:
                    continue
                for c in df.columns:
                    if c not in max_dict:
                        max_dict[c] = df[c].max()
                    else:
                        max_dict[c] = max(df[c].max(), max_dict[c])
                    if c not in min_dict:
                        min_dict[c] = df[c].min()
                    else:
                        min_dict[c] = min(df[c].min(), min_dict[c])
            max_json = json.dumps(max_dict, default=json_type)
            min_json = json.dumps(min_dict, default=json_type)
            out_f = open(self.info_file, 'w')
            out_f.write(max_json + os.linesep + min_json)
        else:
            lines = open(self.info_file, 'r').readlines()
            max_dict = json.loads(lines[0])
            min_dict = json.loads(lines[1])

        self.column_max = max_dict
        self.column_min = min_dict

        # label max/min value
        self.label_max = self.column_max[self.label]
        self.label_min = self.column_min[self.label]
        logging.info("label: %d-%d" % (self.label_min, self.label_max))

        # #users, # items
        self.user_num, self.item_num = 0, 0
        if 'uid' in self.column_max:
            self.user_num = self.column_max['uid'] + 1
        if 'iid' in self.column_max:
            self.item_num = self.column_max['iid'] + 1
        logging.info("# of users: %d" % self.user_num)
        logging.info("# of items: %d" % self.item_num)

        self.user_features = [f for f in self.column_max.keys() if f.startswith('u_')]
        logging.info("# of user features: %d" % len(self.user_features))
        self.item_features = [f for f in self.column_max.keys() if f.startswith('i_')]
        logging.info("# of item features: %d" % len(self.item_features))
        self.context_features = [f for f in self.column_max.keys() if f.startswith('c_')]
        logging.info("# of context features: %d" % len(self.context_features))
        self.features = self.context_features + self.user_features + self.item_features
        logging.info("# of features: %d" % len(self.features))

    def _load_his(self):
        """
        load dataset interactions which have been grouped by uid.
        two columns 'uid' and 'iid'.
        :return:
        """
        if not self.load_data:
            return
        if not os.path.exists(self.train_his_file):
            logging.info("building train history csv...")
            train_his_df = group_user_interactions_df(self.train_df, label=self.label, seq_sep=self.seqs_sep)
            train_his_df.to_csv(self.train_his_file, index=False, sep=self.sep)
        if not os.path.exists(self.vt_his_file):
            logging.info("building vt history csv...")
            vt_df = pd.concat([self.validation_df, self.test_df])
            vt_his_df = group_user_interactions_df(vt_df, label=self.label, seq_sep=self.seqs_sep)
            vt_his_df.to_csv(self.vt_his_file, index=False, sep=self.sep)

        def build_his(his_df, seqs_sep):
            uids = his_df['uid'].tolist()
            iids = his_df['iids'].str.split(seqs_sep).values
            # iids = [i.split(self.seqs_sep) for i in his_df['iids'].tolist()]
            iids = [[int(j) for j in i] for i in iids]
            user_his = dict(zip(uids, iids))
            return user_his

        self.train_his_df, self.train_user_his = None, None
        self.vt_his_df, self.vt_user_his = None, None
        if self.load_data:
            logging.info("load history csv...")
            self.train_his_df = pd.read_csv(self.train_his_file, sep=self.sep)
            self.train_user_his = build_his(self.train_his_df, self.seqs_sep)
            self.vt_his_df = pd.read_csv(self.vt_his_file, sep=self.sep)
            self.vt_user_his = build_his(self.vt_his_df, self.seqs_sep)

    def feature_info(self, include_id=True, include_item_features=True, include_user_features=True):
        features = []
        if include_id:
            features.extend(['uid', 'iid'])
        if include_user_features:
            features.extend(self.user_features)
        if include_item_features:
            features.extend(self.item_features)
        feature_dims = 0
        feature_min, feature_max = [], []
        for f in features:
            feature_min.append(feature_dims)
            feature_dims += int(self.column_max[f] + 1)
            feature_max.append(feature_dims - 1)
        logging.info('Model # of features %d' % len(features))
        logging.info('Model # of feature dims %d' % feature_dims)
        return features, feature_dims, feature_min, feature_max

    def append_his(self, last_n=10, supply=True, neg=False, neg_column=True):
        """
        Generate user history interaction sequence.
        Data in train,validation,test must be sorted by timestamp. Train data comes earlier
        than validation and earlier than test.
        :param last_n: the max history length to keep，<=0 value means keeps all.
        :param supply: True for appending -1 to make up the length
        :param neg: if the interactions including negative feedbacks
        :param neg_column: if add a new column for negative sequence，iff neg_include=True is valid.
        :return:
        """
        his_dict, neg_dict = {}, {}
        for df in [self.train_df, self.validation_df, self.test_df]:
            if df is None:
                continue
            history, neg_history = [], []
            uids, iids, labels = df['uid'].tolist(), df['iid'].tolist(), df[self.label].tolist()
            for i, uid in enumerate(uids):
                iid, label = str(iids[i]), labels[i]

                if uid not in his_dict:
                    his_dict[uid] = []
                if uid not in neg_dict:
                    neg_dict[uid] = []

                tmp_his = his_dict[uid] if last_n <= 0 else his_dict[uid][-last_n:]
                tmp_neg = neg_dict[uid] if last_n <= 0 else neg_dict[uid][-last_n:]
                if supply:
                    tmp_his = tmp_his + ['-1'] * last_n
                    tmp_neg = tmp_neg + ['-1'] * last_n
                history.append(','.join(tmp_his[:last_n]))
                neg_history.append(','.join(tmp_neg[:last_n]))
                if label <= 0 and not neg_column and neg:
                    his_dict[uid].append('~' + iid)
                elif label <= 0 and neg_column:
                    neg_dict[uid].append(iid)
                elif label > 0:
                    his_dict[uid].append(iid)
            df[global_p.C_HISTORY] = history
            if neg and neg_column:
                df[global_p.C_HISTORY_NEG] = neg_history

    def drop_neg(self):
        logging.info('Drop Neg Samples...')
        self.train_df = self.train_df[self.train_df[self.label] > 0].reset_index(drop=True)
        self.validation_df = self.validation_df[self.validation_df[self.label] > 0].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df[self.label] > 0].reset_index(drop=True)
        self.train_df[self.label] = 1
        self.validation_df[self.label] = 1
        self.test_df[self.label] = 1
        logging.info("size of train: %d" % len(self.train_df))
        logging.info("size of validation: %d" % len(self.validation_df))
        logging.info("size of test: %d" % len(self.test_df))
