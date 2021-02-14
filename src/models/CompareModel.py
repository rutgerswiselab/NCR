# coding=utf-8

import torch
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils import utils
import numpy as np
from utils import global_p


class CompareModel(BaseModel):
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='CompareModel'):
        parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
        parser.add_argument('--r_weight', type=float, default=0.1,
                            help='Weight of logic regularizer loss')
        parser.add_argument('--ppl_weight', type=float, default=0,
                            help='Weight of uv interaction prediction loss')
        parser.add_argument('--pos_weight', type=float, default=0,
                            help='Weight of positive purchase loss')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 r_weight, ppl_weight, pos_weight, random_seed, model_path):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        self.r_weight = r_weight
        self.ppl_weight = ppl_weight
        self.pos_weight = pos_weight
        self.sim_scale = 10
        BaseModel.__init__(self, label_min=label_min, label_max=label_max,
                           feature_num=feature_num, random_seed=random_seed,
                           model_path=model_path)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 0.1, size=self.ui_vector_size).astype(np.float32)), requires_grad=False)
        self.not_layer_1 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.not_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.and_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.and_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.or_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.or_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        # self.output_1 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size, bias=False)
        # self.output_2 = torch.nn.Linear(self.ui_vector_size, 1, bias=False)

    def logic_not(self, vector):
        vector = F.relu(self.not_layer_1(vector))
        vector = self.not_layer_2(vector)
        return vector

    def logic_and(self, vector1, vector2):
        assert(len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.and_layer_1(vector))
        vector = self.and_layer_2(vector)
        return vector

    def logic_or(self, vector1, vector2):
        assert (len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.or_layer_1(vector))
        vector = self.or_layer_2(vector)
        return vector

    def purchase_gate(self, uv_vector):
        uv_vector = F.relu(self.purchase_layer_1(uv_vector))
        uv_vector = self.purchase_layer_2(uv_vector)
        return uv_vector

    # def logic_output(self, vector):
    def mse(self, vector1, vector2):
        return ((vector1 - vector2) ** 2).mean()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[global_p.C_HISTORY]
        batch_size, his_length = list(history.size())

        history_pos_tag = feed_dict[global_p.C_HISTORY_POS_TAG].unsqueeze(2).float()

        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(u_ids)
        item_vectors = self.iid_embeddings(i_ids)

        # concat iids with uids and send to purchase gate to prepare for logic_and gate
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(history_pos_tag.size(0), history_pos_tag.size(1), uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(history)

        # concatenate user embedding with history item embeddings
        his_vectors = torch.cat((uh_vectors, his_vectors), dim=2)

        # True/False representation of user item interactions
        his_vectors = self.purchase_gate(his_vectors)
        not_his_vectors = self.logic_not(his_vectors)
        his_vectors = history_pos_tag * his_vectors + (1 - history_pos_tag) * not_his_vectors

        constraint = [his_vectors]
        # False representation
        # false = self.logic_not(self.true)

        # false and true matrix by history pos tag
        # tf_matrix = history_pos_tag * self.true + (1 - history_pos_tag) * false

        # purchase interaction prediction (mean of cosine similarity)
        # predict_purchase = torch.mean(F.cosine_similarity(his_vectors, tf_matrix, dim=2))
        # predict_purchase = predict_purchase + torch.mean(
        #     F.cosine_similarity(item_vectors, self.true.expand(item_vectors.size(0), item_vectors.size(1)))
        # )
        # predict_purchase = self.mse(his_vectors, tf_matrix) / (tf_matrix ** 2).mean()
        # predict_purchase = predict_purchase + self.mse(item_vectors, self.true.expand_as(item_vectors)) / \
        #                    (self.true.expand_as(item_vectors) ** 2).mean()

        tmp_vector = his_vectors[:, 0]
        shuffled_history_idx = [i for i in range(1, his_length)]
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_and(tmp_vector, his_vectors[:, i])
            constraint.append(tmp_vector.view(batch_size, -1, self.ui_vector_size))
        # left_vector = self.logic_not(tmp_vector)
        left_vector = tmp_vector
        constraint.append(left_vector.view(batch_size, -1, self.ui_vector_size))

        right_vector = self.logic_not(item_vectors)
        constraint.append(right_vector.view(batch_size, -1, self.ui_vector_size))
        sent_vector = self.logic_or(left_vector, right_vector)
        constraint.append(sent_vector.view(batch_size, -1, self.ui_vector_size))
        # check_list.append(('sent_vector', sent_vector))
        if feed_dict['rank'] == 1:
            prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * 10
        else:
            prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * \
                         (self.label_max - self.label_min) / 2 + (self.label_max + self.label_min) / 2
        # check_list.append(('prediction', prediction))
        # prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        constraint = torch.cat(constraint, dim=1)
        out_dict = {'prediction': prediction,
                    # 'predict_purchase': predict_purchase,
                    # 'his_vectors': his_vectors,
                    'check': check_list,
                    'constraint': constraint}
        return out_dict

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        check_list = out_dict['check']
        # predict_purchase = out_dict['predict_purchase']
        false = self.logic_not(self.true).view(1, -1)
        # his_vectors = out_dict['his_vectors']
        constraint = out_dict['constraint']

        # regularizer
        dim = len(constraint.size())-1

        # length constraint
        # r_length = constraint.norm(dim=dim)()

        # not
        # r_not_true = self.mse(self.logic_not(self.logic_not(self.true)), self.true) / (self.true ** 2).mean()
        # r_not_true = (1 - (F.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, dim=0)
        #               * self.sim_scale).sigmoid()).sum()
        r_not_not_true = (1 - F.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, dim=0)).sum()
        check_list.append(('r_not_not_true', r_not_not_true))
        # r_not_self = self.mse(self.logic_not(self.logic_not(constraint)), constraint) / (constraint ** 2).mean()
        # r_not_self = (F.cosine_similarity(
        #     self.logic_not(self.logic_not(constraint)), constraint, dim=dim)
        #               * self.sim_scale).sigmoid().mean()
        r_not_not_self = \
            (1 - F.cosine_similarity(self.logic_not(self.logic_not(constraint)), constraint, dim=dim)).mean()
        check_list.append(('r_not_not_self', r_not_not_self))
        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraint), constraint, dim=dim)).mean()

        # and
        # r_and_true = self.mse(
        #     self.logic_and(constraint, self.true.expand_as(constraint)), constraint) / (constraint ** 2).mean()
        # r_and_true = (-F.cosine_similarity(
        #     self.logic_and(constraint, self.true.expand_as(constraint)), constraint, dim=dim)
        #               * self.sim_scale).sigmoid().mean()
        r_and_true = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.true.expand_as(constraint)), constraint, dim=dim)).mean()
        check_list.append(('r_and_true', r_and_true))
        # r_and_false = self.mse(self.logic_and(constraint, false.expand_as(constraint)), false) / (false ** 2).mean()
        # r_and_false = (-F.cosine_similarity(
        #     self.logic_and(constraint, false.expand_as(constraint)), false.expand_as(constraint), dim=dim)
        #                * self.sim_scale).sigmoid().mean()
        r_and_false = (1 - F.cosine_similarity(
            self.logic_and(constraint, false.expand_as(constraint)), false.expand_as(constraint), dim=dim)).mean()
        check_list.append(('r_and_false', r_and_false))
        # r_and_self = self.mse(self.logic_and(constraint, constraint), constraint) / (constraint ** 2).mean()
        r_and_self = (1 - F.cosine_similarity(self.logic_and(constraint, constraint), constraint, dim=dim)).mean()
        check_list.append(('r_and_self', r_and_self))

        # NEW ADDED REG NEED TO TEST
        r_and_not_self = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.logic_not(constraint)), false.expand_as(constraint), dim=dim)).mean()
        check_list.append(('r_and_not_self', r_and_not_self))
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_and(self.logic_not(constraint), constraint), false.expand_as(constraint), dim=dim)).mean()
        check_list.append(('r_and_not_self_inverse', r_and_not_self_inverse))

        # or
        # r_or_true = self.mse(
        #     self.logic_or(constraint, self.true.expand_as(constraint)), self.true) / (self.true ** 2).mean()
        # r_or_true = (-F.cosine_similarity(
        #     self.logic_or(constraint, self.true.expand_as(constraint)), self.true.expand_as(constraint), dim=dim)
        #              * self.sim_scale).sigmoid().mean()
        r_or_true = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.true.expand_as(constraint)), self.true.expand_as(constraint), dim=dim))\
            .mean()
        check_list.append(('r_or_true', r_or_true))
        # r_or_false = self.mse(
        #     self.logic_or(constraint, false.expand_as(constraint)), constraint) / (constraint ** 2).mean()
        # r_or_false = (-F.cosine_similarity(self.logic_or(constraint, false.expand_as(constraint)), constraint, dim=dim)
        #               * self.sim_scale).sigmoid().mean()
        r_or_false = (1 - F.cosine_similarity(
            self.logic_or(constraint, false.expand_as(constraint)), constraint, dim=dim)).mean()
        check_list.append(('r_or_false', r_or_false))
        # r_or_self = self.mse(self.logic_or(constraint, constraint), constraint) / (constraint ** 2).mean()
        # r_or_self = (-F.cosine_similarity(self.logic_or(constraint, constraint), constraint, dim=dim)
        #              * self.sim_scale).sigmoid().mean()
        r_or_self = (1 - F.cosine_similarity(self.logic_or(constraint, constraint), constraint, dim=dim)).mean()
        check_list.append(('r_or_self', r_or_self))

        r_or_not_self = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.logic_not(constraint)), self.true.expand_as(constraint), dim=dim)).mean()
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_or(self.logic_not(constraint), constraint), self.true.expand_as(constraint), dim=dim)).mean()
        check_list.append(('r_or_not_self', r_or_not_self))
        check_list.append(('r_or_not_self_inverse', r_or_not_self_inverse))

        # True/False
        true_false = 1 + F.cosine_similarity(self.true, false.view(-1), dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse + \
                 r_or_true + r_or_false + r_or_self + true_false + r_or_not_self + r_or_not_self_inverse
        r_loss = r_loss * self.r_weight

        # pos_loss = None
        # recommendation loss
        if feed_dict['rank'] == 1:
            # 计算topn推荐的loss，batch前一半是正例，后一半是负例
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            # tf_matrix = self.true.view(1, -1).expand(batch_size, -1)
            pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
            # pos_loss = 10 - torch.mean(pos)
            loss = -(pos - neg).sigmoid().log().sum()
            check_list.append(('bpr_loss', loss))
        else:
            # 计算rating/clicking预测的loss，默认使用mse
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])

        # predict_purchase_loss = (2 - predict_purchase)
        loss = loss + r_loss #+ self.ppl_weight * predict_purchase_loss #+ self.pos_weight * pos_loss
        check_list.append(('r_loss', r_loss))
        out_dict['loss'] = loss
        out_dict['check'] = check_list
        return out_dict
