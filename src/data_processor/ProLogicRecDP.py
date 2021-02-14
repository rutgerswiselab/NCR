# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from data_processor.HisDataProcessor import HisDataProcessor
from data_processor.DataProcessor import DataProcessor
from utils import global_p


class ProLogicRecDP(HisDataProcessor):
    # data dict中存储模型所需特征信息的key（负例data需要append在最后）
    data_columns = ['X', global_p.C_HISTORY, global_p.C_HISTORY_POS_TAG, global_p.C_HISTORY_LENGTH]

    def format_data_dict(self, df):
        """
        除了常规的uid,iid,label,user、item、context特征外，还需处理历史交互
        :param df: 训练、验证、测试df
        :return:
        """
        his_list = df[global_p.C_HISTORY].apply(lambda x: x.split(','))
        his_length = his_list.apply(lambda x: 0 if x[0] == '' else len(x))

        # 去掉历史记录长度为0的条目
        his_length = his_length[his_length > 0]
        df, his_list = df.loc[his_length.index], his_list.loc[his_length.index]

        data_dict = DataProcessor.format_data_dict(self, df)

        history_pos_tag = his_list.apply(lambda x: [0 if i.startswith('~') else 1 for i in x])
        history = his_list.apply(lambda x: [int(i[1:]) if i.startswith('~') else int(i) for i in x])

        data_dict[global_p.C_HISTORY] = history.values
        data_dict[global_p.C_HISTORY_POS_TAG] = history_pos_tag.values
        data_dict[global_p.C_HISTORY_LENGTH] = np.array([len(h) for h in data_dict[global_p.C_HISTORY]])
        return data_dict

    def get_boolean_test_data(self):
        logging.info('Prepare Boolean Test Data...')
        df = self.data_loader.test_df
        self.boolean_test_data = self.format_data_dict(df)
        self.boolean_test_data[global_p.K_SAMPLE_ID] = np.arange(0, len(self.boolean_test_data['Y']))
        return self.boolean_test_data