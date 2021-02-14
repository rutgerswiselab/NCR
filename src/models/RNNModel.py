# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils import utils
import numpy as np
from utils import global_p


class RNNModel(BaseModel):
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='RNNModel'):
        parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                random_seed, model_path):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        self.sim_scale = 10
        BaseModel.__init__(self, label_min=label_min, label_max=label_max,
                           feature_num=feature_num, random_seed=random_seed,
                           model_path=model_path)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.gru = nn.GRU(2 * self.ui_vector_size, self.ui_vector_size, batch_first=True)
        self.output_layer = nn.Linear(self.ui_vector_size, 1)
        # self.h0_embed = torch.nn.Embedding(1, self.ui_vector_size)

    def mse(self, vector1, vector2):
        return ((vector1 - vector2) ** 2).mean()

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[global_p.C_HISTORY]
        batch_size, his_length = list(history.size())

        # user/item vectors shape: (batch_size, embedding_size)
        user_vectors = self.uid_embeddings(u_ids)
        item_vectors = self.iid_embeddings(i_ids)

        # expand user vector to prepare for concatenating with history item vectors
        uh_vectors = user_vectors.view(batch_size, 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(batch_size, his_length + 1, uh_vectors.size(2))

        # history item purchase hidden factors shape: (batch, user, embedding)
        his_vectors = self.iid_embeddings(history)
        his_vectors = torch.cat((his_vectors, item_vectors.view(batch_size, -1, item_vectors.size(1))), dim=1)

        # concatenate user embedding with history item embeddings
        his_vectors = torch.cat((uh_vectors, his_vectors), dim=2)

        # h0 = self.h0_embed(torch.LongTensor([0]).cuda())
        # h0 = h0.repeat(1, batch_size, 1)
        _, hidden = self.gru(his_vectors, None)
        hidden = hidden.view(batch_size, self.ui_vector_size)

        prediction = self.output_layer(hidden)
        out_dict = {'prediction': prediction,
                    # 'predict_purchase': predict_purchase,
                    # 'his_vectors': his_vectors,
                    'check': check_list
                    }
        return out_dict

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        check_list = out_dict['check']

        # recommendation loss
        if feed_dict['rank'] == 1:
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            # tf_matrix = self.true.view(1, -1).expand(batch_size, -1)
            pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
            loss = -(pos - neg).sigmoid().log().sum()
            check_list.append(('bpr_loss', loss))
        else:
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])

        out_dict['loss'] = loss
        out_dict['check'] = check_list
        return out_dict
