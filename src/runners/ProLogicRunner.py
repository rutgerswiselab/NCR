# coding=utf-8

import torch.nn.functional as F
import logging
import torch
import pickle
from utils import utils, global_p
from tqdm import tqdm
import numpy as np
import copy
from sklearn.metrics import *
import itertools
import pandas as pd
from runners.BaseRunner import BaseRunner


class ProLogicRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-4,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, metrics='RMSE', check_epoch=10, early_stop=1):
        """
        初始化
        :param optimizer: 优化器名字
        :param learning_rate: 学习率
        :param epoch: 总共跑几轮
        :param batch_size: 训练batch大小
        :param eval_batch_size: 测试batch大小
        :param dropout: dropout比例
        :param l2: l2权重
        :param metrics: 评价指标，逗号分隔
        :param check_epoch: 每几轮输出check一次模型中间的一些tensor
        :param early_stop: 是否自动提前终止训练
        """
        BaseRunner.__init__(self, optimizer=optimizer, learning_rate=learning_rate,
                            epoch=epoch, batch_size=batch_size, eval_batch_size=eval_batch_size,
                            dropout=dropout, l2=l2, metrics=metrics,
                            check_epoch=check_epoch, early_stop=early_stop)

    def accuracy_calc(self, p, l):
        """
        calculate the accuracy with each bit flip
        :param p: predicted value
        :param l: ground truth value calculated by expression_evaluator
        :return: accuracy rate
        """
        return accuracy_score(l, p)

    def _data_reformat(self, data, bit_reverse_indices):
        """
        update the x_tag
        :param data: data dictionary
        :param bit_reverse_indices: a list with the indices of the bit to be reversed
        :return:
        """
        new_data = copy.deepcopy(data)
        for tag in new_data[global_p.C_HISTORY_POS_TAG]:
            for index in bit_reverse_indices:
                tag[index] = 1 - tag[index]
        return new_data

    def _boolean_evaluate(self, model, data, data_processor, bit_reverse_index):
        new_data = self._data_reformat(data, bit_reverse_index)
        batches = data_processor.prepare_batches(new_data, self.eval_batch_size, train=False)
        batches = self.batches_add_control(batches, train=False)

        predictions = []
        interims = []
        model.eval()
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            result = model.predict(batch)
            prediction = result['prediction']
            interim = result['interim']
            interims.append(interim.detach())
            predictions.append(prediction.detach())
        predictions = np.concatenate(predictions)
        interims = np.concatenate(interims, axis=0)
        sample_ids = np.concatenate([b[global_p.K_SAMPLE_ID] for b in batches])

        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[global_p.K_SAMPLE_ID]])
        reorder_dict_2 = dict(zip(sample_ids, interims))

        interims = np.array([reorder_dict_2[i] for i in data[global_p.K_SAMPLE_ID]])
        return predictions, interims

    @staticmethod
    def _enum_subsets(input_set):
        """
        enumerate all the subsets of given input_set
        return: a dictionary with key for the number of elements in the subsets and
        value is a list of elements
        """
        result_dict = {}
        for i in range(1, len(input_set) + 1):
            tmp_list = list(map(list, itertools.combinations(input_set, i)))
            result_dict[i] = tmp_list
        return result_dict

    @staticmethod
    def _gen_prediction_dict(p, data):
        df = pd.DataFrame()
        df['uid'] = data['uid']
        df['iid'] = data['iid']
        df['p'] = p
        df = df.sort_values(by='p', ascending=False)
        df_group = df.groupby('uid')
        y_dict = {}
        for uid, group in df_group:
            tmp_iid = group['iid'].tolist()[:1][0]
            y_dict[uid] = tmp_iid
        return y_dict

    @staticmethod
    def _accuracy_calc_from_dict(original_dict, updated_dict):
        assert len(original_dict) == len(updated_dict)
        counter = 0
        for key in original_dict:
            if updated_dict[key] == original_dict[key]:
                counter += 1
        return counter, len(original_dict)

    @staticmethod
    def _statistic_info(data):
        path = './ml100k_freq_info.pkl'
        with open(path, 'rb') as file:
            item_dict = pickle.load(file)
        tmp_list = []
        for key in data:
            tmp_list.append(item_dict[data[key]])
        tmp_list = np.array(tmp_list)
        logging.info('\n average frequency: %.1f' % tmp_list.mean())
        logging.info('\n max frequency: %.1f' % tmp_list.max())
        logging.info('\n min frequency: %.1f' % tmp_list.min())

    @staticmethod
    def _statistic_of_difference(original, updated):
        path = './ml100k_freq_info.pkl'
        with open(path, 'rb') as file:
            item_dict = pickle.load(file)
        unchanged_dict = {}
        changed_dict = {}
        for key in original:
            if original[key] == updated[key]:
                unchanged_dict[original[key]] = item_dict[original[key]]
            else:
                changed_dict[key] = {original[key]: item_dict[original[key]], updated[key]: item_dict[updated[key]]}
        unchanged_freq_max = max(unchanged_dict, key=unchanged_dict.get)
        unchanged_freq_min = min(unchanged_dict, key=unchanged_dict.get)
        unchanged_freq_mean = np.array([unchanged_dict[k] for k in unchanged_dict]).mean()
        logging.info("unchanged_freq_max: {}".format(unchanged_dict[unchanged_freq_max]))
        logging.info("unchanged_freq_min: {}".format(unchanged_dict[unchanged_freq_min]))
        logging.info("unchanged_freq_mean: {}".format(unchanged_freq_mean))

        return unchanged_dict, changed_dict

    def boolean_test(self, model, data, data_processor):
        """
        reverse bits to test the boolean sensitivity
        :param model: model name
        :param data: data to use
        :param data_processor: data processor
        :return:
        """
        length_dict = {}
        lengths = [len(x) for x in data[global_p.C_HISTORY]]
        for idx, l in enumerate(lengths):
            if l not in length_dict:
                length_dict[l] = []
            length_dict[l].append(idx)
        lengths = list(length_dict.keys())
        # for key in lengths:
        #     print('{}: {}'.format(key, len(length_dict[key])))

        # accumulate_accuracy = 0.
        result_dict = {}    # store the accuracy of given number of bits are reversed.
        counter_dict = {}
        info_dict = {}
        for l in tqdm(lengths, leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            rows = length_dict[l]
            tmp_data = {}
            for key in data:
                if data[key].dtype == np.object:
                    tmp_data[key] = np.array([np.array(data[key][r]) for r in rows])
                    # tmp_data['id'] = np.array([r] for r in rows)
                else:
                    tmp_data[key] = data[key][rows]
                    # tmp_data['id'] = np.array([r] for r in rows)
            expression_length = len(tmp_data[global_p.C_HISTORY][0])
            index_set = [i for i in range(expression_length)]
            index_sets_dict = self._enum_subsets(index_set)

            tmp_interim = None
            for key in index_sets_dict:
                acc_counter = 0
                acc_len = 0
                acc_sim = 0
                sim_counter = 0
                for index_list in index_sets_dict[key]:
                    p = self.predict(model, tmp_data, data_processor)
                    original_predict = self._gen_prediction_dict(p, tmp_data)

                    predictions, interims = self._boolean_evaluate(model, tmp_data, data_processor, index_list)
                    updated_predict = self._gen_prediction_dict(predictions, tmp_data)

                    if tmp_interim is None:
                        tmp_interim = copy.deepcopy(interims)
                    else:
                        acc_sim += F.cosine_similarity(torch.from_numpy(tmp_interim), torch.from_numpy(interims), dim=-1).mean()

                        tmp_interim = copy.deepcopy(interims)
                        sim_counter += 1

                    self._statistic_info(original_predict)
                    unchanged_dict, changed_dict = self._statistic_of_difference(original_predict, updated_predict)
                    # print(unchanged_dict)
                    # print(changed_dict)
                    print(asasd)
                    tmp_counter, tmp_len = self._accuracy_calc_from_dict(original_predict, updated_predict)
                    acc_counter += tmp_counter
                    acc_len += tmp_len

                    tmp_str = ' '.join([str(e) for e in index_list])
                    if tmp_str not in info_dict:
                        info_dict[tmp_str] = tmp_counter / tmp_len

                accuracy = acc_counter / acc_len
                similarity = acc_sim / sim_counter
                if key not in result_dict:
                    result_dict[key] = {'accuracy': accuracy, 'similarity': similarity}
                    counter_dict[key] = 1
                else:
                    result_dict[key]['accuracy'] += accuracy
                    result_dict[key]['similarity'] += similarity
                    counter_dict[key] += 1
                # accumulate_accuracy += (accuracy / expression_length)
        for key in result_dict:
            logging.info(
                '{} bit reverse average accuracy: {}\taverage similarity: {}'.format(
                    str(key),
                    result_dict[key]['accuracy'] / counter_dict[key],
                    result_dict[key]['similarity'] / counter_dict[key]
                )
            )
        logging.info('----------- Details ------------')
        for key in info_dict:
            logging.info(str(key) + ': ' + str(info_dict[key]))