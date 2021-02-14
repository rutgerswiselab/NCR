# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from utils import global_p


class DataProcessor(object):
    data_columns = ['X']  # input feature column (include uid, iid)

    @staticmethod
    def parse_dp_args(parser):
        """
        parse data processor related command line arguments
        """
        parser.add_argument('--test_neg_n', type=int, default=10,
                            help='Negative sample num for each instance in test/validation set.')
        return parser

    def __init__(self, data_loader, model, rank, test_neg_n):
        """
        Initialization
        :param data_loader: DataLoader object
        :param model: Model object
        :param rank: 1=ranking, 0=rating prediction
        :param test_neg_n: ranking negative sample rate, pos:neg=1:test_neg_n
        """
        self.data_loader = data_loader
        self.model = model
        self.rank = rank
        self.train_data, self.validation_data, self.test_data = None, None, None

        self.test_neg_n = test_neg_n

        if self.rank == 1:
            # 生成用户交互的字典，方便采样负例时查询，不要采到正例
            self.train_history_dict = defaultdict(set)
            for uid in data_loader.train_user_his.keys():
                self.train_history_dict[uid] = set(data_loader.train_user_his[uid])
            # print(self.train_history_dict[405])
            self.vt_history_dict = defaultdict(set)
            for uid in data_loader.vt_user_his.keys():
                self.vt_history_dict[uid] = set(data_loader.vt_user_his[uid])
            # print(self.vt_history_dict[405])

        self.vt_batches_buffer = {}

    def get_train_data(self, epoch):
        if self.train_data is None or epoch < 0:
            logging.info('Prepare Train Data...')
            self.train_data = self.format_data_dict(self.data_loader.train_df)
            self.train_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.train_data['Y']))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
        return self.train_data

    def get_validation_data(self):
        if self.validation_data is None:
            logging.info('Prepare Validation Data...')
            df = self.data_loader.validation_df
            if self.rank == 1:
                neg_df = self.generate_neg_df(
                    uid_list=df['uid'].tolist(), iid_list=df['iid'].tolist(),
                    df=df, neg_n=self.test_neg_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.validation_data = self.format_data_dict(df)
            self.validation_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.validation_data['Y']))
        return self.validation_data

    def get_test_data(self):
        if self.test_data is None:
            logging.info('Prepare Test Data...')
            df = self.data_loader.test_df
            if self.rank == 1:
                neg_df = self.generate_neg_df(
                    uid_list=df['uid'].tolist(), iid_list=df['iid'].tolist(),
                    df=df, neg_n=self.test_neg_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.test_data = self.format_data_dict(df)
            self.test_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.test_data['Y']))
        return self.test_data

    def get_train_batches(self, batch_size, epoch):
        return self.prepare_batches(self.get_train_data(epoch), batch_size, train=True)

    def get_validation_batches(self, batch_size):
        return self.prepare_batches(self.get_validation_data(), batch_size, train=False)

    def get_test_batches(self, batch_size):
        return self.prepare_batches(self.get_test_data(), batch_size, train=False)

    def _get_feed_dict_rt(self, data, batch_start, batch_size, train):
        batch_end = min(len(data['X']), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        feed_dict = {'train': train, 'rank': 0,
                     global_p.K_SAMPLE_ID: data[global_p.K_SAMPLE_ID][batch_start:batch_start + real_batch_size]}
        if 'Y' in data:
            feed_dict['Y'] = utils.numpy_to_torch(data['Y'][batch_start:batch_start + real_batch_size])
        else:
            feed_dict['Y'] = utils.numpy_to_torch(np.zeros(shape=real_batch_size))
        for c in self.data_columns:
            feed_dict[c] = utils.numpy_to_torch(
                data[c][batch_start: batch_start + real_batch_size])
        return feed_dict

    def _get_feed_dict_rk(self, data, batch_start, batch_size, train, neg_data=None):
        if not train:
            feed_dict = self._get_feed_dict_rt(
                data=data, batch_start=batch_start, batch_size=batch_size, train=train)
            feed_dict['rank'] = 1
        else:
            batch_end = min(len(data['X']), batch_start + batch_size)
            real_batch_size = batch_end - batch_start
            neg_columns_dict = {}
            if neg_data is None:
                logging.warning('neg_data is None')
                neg_df = self.generate_neg_df(
                    uid_list=data['uid'][batch_start: batch_start + real_batch_size],
                    iid_list=data['iid'][batch_start: batch_start + real_batch_size],
                    df=self.data_loader.train_df, neg_n=1, train=True)
                neg_data = self.format_data_dict(neg_df)
                for c in self.data_columns:
                    neg_columns_dict[c] = neg_data[c]
            else:
                for c in self.data_columns:
                    neg_columns_dict[c] = neg_data[c][batch_start: batch_start + real_batch_size]
            y = np.concatenate([np.ones(shape=real_batch_size, dtype=np.float32),
                                np.zeros(shape=real_batch_size, dtype=np.float32)])
            sample_id = data[global_p.K_SAMPLE_ID][batch_start:batch_start + real_batch_size]
            neg_sample_id = sample_id + len(self.train_data['Y'])
            feed_dict = {
                'train': train, 'rank': 1,
                'Y': utils.numpy_to_torch(y),
                global_p.K_SAMPLE_ID: np.concatenate([sample_id, neg_sample_id])}
            for c in self.data_columns:
                feed_dict[c] = utils.numpy_to_torch(
                    np.concatenate([data[c][batch_start: batch_start + real_batch_size], neg_columns_dict[c]]))
        return feed_dict

    def _prepare_batches_rt(self, data, batch_size, train):
        """
        for rating/clicking prediction
        """
        if data is None:
            return None
        num_example = len(data['X'])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self._get_feed_dict_rt(data, batch * batch_size, batch_size, train))
        return batches

    def _prepare_batches_rk(self, data, batch_size, train):
        """
        for ranking task
        """
        if data is None:
            return None
        num_example = len(data['X'])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        # 如果是训练，则需要对对应的所有正例采一个负例
        neg_data = None
        if train:
            neg_df = self.generate_neg_df(
                uid_list=data['uid'], iid_list=data['iid'],
                df=self.data_loader.train_df, neg_n=1, train=True)
            neg_data = self.format_data_dict(neg_df)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self._get_feed_dict_rk(data, batch * batch_size, batch_size, train, neg_data))
        return batches

    def prepare_batches(self, data, batch_size, train):
        """
        convert data dict to batches
        :param data: dict generated by self.get_*_data() and self.format_data_dict()
        :param batch_size: batch size
        :param train: train or validation/test
        :return: list of batches
        """
        buffer_key = ''
        if data is self.validation_data:
            buffer_key = 'validation_' + str(batch_size)
        elif data is self.test_data:
            buffer_key = 'test_' + str(batch_size)
        if buffer_key in self.vt_batches_buffer:
            return self.vt_batches_buffer[buffer_key]

        if self.rank == 1:
            batches = self._prepare_batches_rk(data=data, batch_size=batch_size, train=train)
        else:
            batches = self._prepare_batches_rt(data=data, batch_size=batch_size, train=train)

        if buffer_key != '':
            self.vt_batches_buffer[buffer_key] = batches
        return batches

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None):
        """
        :param data: data dict，generated by self.get_*_data() and self.format_data_dict()
        :param batch_start: start index of each batch
        :param batch_size: batch size
        :param train: train or validation/test
        :param neg_data: negative sample data dictionary
        :return: feed dict
        :return:
        """
        if self.rank == 1:
            return self._get_feed_dict_rk(data=data, batch_start=batch_start, batch_size=batch_size, train=train,
                                          neg_data=neg_data)
        return self._get_feed_dict_rt(data=data, batch_start=batch_start, batch_size=batch_size, train=train)

    def format_data_dict(self, df):
        """
        format Dataframe to data dictionary
        :param df: pandas Dataframe, contains 'uid','iid','label' three columns (at least)
        :return: data dict
        """

        data_loader, model = self.data_loader, self.model
        data = {}
        # record uid, iid
        out_columns = []
        if 'uid' in df:
            out_columns.append('uid')
            data['uid'] = df['uid'].values
        if 'iid' in df:
            out_columns.append('iid')
            data['iid'] = df['iid'].values

        # label info is in 'Y' column
        if data_loader.label in df.columns:
            data['Y'] = np.array(df[data_loader.label], dtype=np.float32)
        else:
            logging.warning('No Labels In Data: ' + data_loader.label)
            data['Y'] = np.zeros(len(df), dtype=np.float32)

        ui_id = df[out_columns]

        out_df = ui_id
        if data_loader.user_df is not None and model.include_user_features:
            out_columns.extend(data_loader.user_features)
            out_df = pd.merge(out_df, data_loader.user_df, on='uid', how='left')
        if data_loader.item_df is not None and model.include_item_features:
            out_columns.extend(data_loader.item_features)
            out_df = pd.merge(out_df, data_loader.item_df, on='iid', how='left')
        out_df = out_df.fillna(0)

        if model.include_context_features:
            context = df[data_loader.context_features]
            out_df = pd.concat([out_df, context], axis=1, ignore_index=True)

        if not model.include_id:
            out_df = out_df.drop(columns=['uid', 'iid'])

        base = 0
        for feature in out_df.columns:
            out_df[feature] = out_df[feature].apply(lambda x: x + base)
            base += int(data_loader.column_max[feature] + 1)

        if model.append_id:
            x = pd.concat([ui_id, out_df], axis=1, ignore_index=True)
            data['X'] = x.values.astype(int)
        else:
            data['X'] = out_df.values.astype(int)
        # print(data['X'].shape)
        assert len(data['X']) == len(data['Y'])
        return data

    def generate_neg_df(self, uid_list, iid_list, df, neg_n, train):
        """
        Generate negative samples
        :param uid_list: users who need to get negative samples
        :param iid_list: users observed interactions
        :param df: dataframe information
        :param neg_n: number of negative samples
        :param train: sample for train or validation/test
        :return:
        """
        neg_df = self._sample_neg_from_uid_list(
            uids=uid_list, neg_n=neg_n, train=train, other_infos={'iid': iid_list})
        neg_df = pd.merge(neg_df, df, on=['uid', 'iid'], how='left')
        neg_df = neg_df.drop(columns=['iid'])
        neg_df = neg_df.rename(columns={'iid_neg': 'iid'})
        neg_df = neg_df[df.columns]
        neg_df[self.data_loader.label] = 0
        return neg_df

    def _sample_neg_from_uid_list(self, uids, neg_n, train, other_infos=None):
        """
        Get negative samples based on user history
        :param uids: uid list
        :param neg_n: the number of negative samples
        :param train: sample for train data or validation/testing
        :param other_infos: other than uid,iid,label，history interactions are included here
        :return: DataFrame, which needs self.format_data_dict() to convert to data dictionary
        """
        if other_infos is None:
            other_infos = {}
        uid_list, iid_list = [], []

        other_info_list = {}
        for info in other_infos:
            other_info_list[info] = []

        # record sampled iids to get avoid of sampled items to be selected again
        tmp_history_dict = defaultdict(set)
        item_num = self.data_loader.item_num
        for index, uid in enumerate(uids):
            if train:
                inter_iids = self.train_history_dict[uid] | tmp_history_dict[uid]
            else:
                inter_iids = self.train_history_dict[uid] | self.vt_history_dict[uid] | tmp_history_dict[uid]

            # check total number of negative samples that are available
            remain_iids_num = item_num - len(inter_iids)
            remain_iids = None
            # use np.choice if candidates are not much
            if 1.0 * remain_iids_num / item_num < 0.2:
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids]
            # check if we have enough negative samples
            assert remain_iids_num >= neg_n
            if remain_iids is None:
                for i in range(neg_n):
                    iid = np.random.randint(1, self.data_loader.item_num)
                    while iid in inter_iids:
                        iid = np.random.randint(1, self.data_loader.item_num)
                    uid_list.append(uid)
                    iid_list.append(iid)
                    tmp_history_dict[uid].add(iid)
            else:
                iids = np.random.choice(remain_iids, neg_n, replace=False)
                uid_list.extend([uid] * neg_n)
                iid_list.extend(iids)
                tmp_history_dict[uid].update(iids)
            for info in other_infos:
                other_info_list[info].extend([other_infos[info][index]] * neg_n)
        neg_df = pd.DataFrame(data=list(zip(uid_list, iid_list)), columns=['uid', 'iid_neg'])
        for info in other_infos:
            neg_df[info] = other_info_list[info]

        return neg_df
