# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils import global_p


class SVDPP(RecModel):
    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.iid_embeddings_implicit = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[global_p.C_HISTORY]
        his_vector = history.mm(self.iid_embeddings_implicit.weight)

        # normalize
        his_length = feed_dict[global_p.C_HISTORY_LENGTH]
        valid_his = his_length.gt(0).float()
        tmp_length = his_length.float() * valid_his + (1 - valid_his) * 1
        his_vector = his_vector / tmp_length.sqrt().view([-1, 1])

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        prediction = ((cf_u_vectors + his_vector) * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias

        out_dict = {'prediction': prediction,
                    'check': check_list}
        return out_dict
