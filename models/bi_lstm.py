import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import sys

seed_num = 223
torch.manual_seed(seed_num)
random.seed(seed_num)

import pickle

"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()

        self.embedding_init_path = args.embedding_init_path

        self.hidden_dim = args.num_lstm_units

        V = args.vocab_size
        D = args.embedding_size

        # word embedding
        self.embed = nn.Embedding(V, D, padding_idx=0)

        # for p in self.parameters():
        #     p.requires_grad = False

        self.dropout_embed = nn.Dropout(0.25)

        self.bilstm = nn.ModuleList()
        self.bilstm.append(nn.LSTM(D, self.hidden_dim, num_layers=2, dropout=0.0, bidirectional=True, bias=False))



    def forward(self, text, text_length):

        # print(text)

        embed = self.embed(text)

        text_length = text_length

        bilstm_out = self.bilstm_out(embed, text_length, 0)

        bilstm_out_conp = bilstm_out.new(bilstm_out.shape[0], 100-bilstm_out.shape[1], bilstm_out.shape[2]).fill_(0)
        bilstm_out = torch.cat((bilstm_out, bilstm_out_conp), 1)
        sentence_embedding_s = bilstm_out

        # bilstm_out, idx = torch.max(bilstm_out, dim=1)
        #
        # bilstm_out = bilstm_out.unsqueeze(2).unsqueeze(2)

        # return bilstm_out, sentence_embedding_s

        return sentence_embedding_s, embed


    def bilstm_out(self, embed, text_length, index):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort].type(torch.int64)

        # print(embed_sort)
        # sys.exit()


        pack = nn.utils.rnn.pack_padded_sequence(embed_sort, length_list.cpu(), batch_first=True)

        bilstm_sort_out, _ = self.bilstm[index](pack)


        # print("bilstm_sort_out: " + str(bilstm_sort_out.shape))


        bilstm_sort_out  = nn.utils.rnn.pad_packed_sequence(bilstm_sort_out, batch_first=True)

        # print("bilstm_sort_out++: " + str(bilstm_sort_out.shape))


        bilstm_sort_out = bilstm_sort_out[0]

        bilstm_out = bilstm_sort_out.index_select(0, idx_unsort)

        return bilstm_out

    def weight_init(self, m):

        self.embedding_init()
        # self.weight.data.normal_(0, 1)
        # nn.init.xavier_normal_(self.embed.weight.data, 1)
        nn.init.xavier_normal_(self.bilstm[0].all_weights[0][0], 1)
        nn.init.xavier_normal_(self.bilstm[0].all_weights[0][1], 1)

    def embedding_init(self):
        save_path = self.embedding_init_path
        with open(save_path, 'rb') as f:
            embeddings = pickle.load(f)
        embeddings = torch.from_numpy(embeddings).float()
        self.embed.weight.data.copy_(embeddings)

