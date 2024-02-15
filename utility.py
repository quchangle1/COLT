#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class sceneTrainDataset(Dataset):
    def __init__(self, conf, s_t_pairs, q_s_pairs, q_t_graph, q_s_graph, num_tools, q_s_for_neg_sample, s_s_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.q_s_pairs = q_s_pairs
        self.q_t_graph = q_t_graph
        self.s_t_pairs = s_t_pairs
        self.q_s_graph = q_s_graph
        self.num_tools = num_tools
        self.neg_sample = neg_sample

        self.q_s_for_neg_sample = q_s_for_neg_sample
        self.s_s_for_neg_sample = s_s_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        query_t, pos_scene = self.q_s_pairs[index]
        label=[]
        all_scenes = [pos_scene]
        s_t_dict = {}
        for item in self.s_t_pairs:
            key = str(item[0])
            value = item[1]
            if key in s_t_dict:
                s_t_dict[key].append(value)
            else:
                s_t_dict[key] = [value]
        all_tools = s_t_dict[str(pos_scene)]
        for i in range (len(all_tools)):
            label.append(1.0)
        while True:
            i = np.random.randint(self.num_tools)
            if self.q_t_graph[query_t, i] == 0 and not i in all_tools:                                                          
                all_tools.append(i)         
                label.append(0.0)                                                                                          
                if len(all_tools) == self.neg_sample:                                                                               
                    break                                                                                                               

        return torch.LongTensor([query_t]), torch.LongTensor(all_scenes), torch.LongTensor(all_tools), torch.LongTensor(label)


    def __len__(self):
        return len(self.q_s_pairs)


class sceneTestDataset(Dataset):
    def __init__(self, q_s_pairs, q_t_graph, num_queries, num_scenes):
        self.q_s_pairs = q_s_pairs
        self.q_t_graph = q_t_graph
        self.num_queries = num_queries
        self.num_scenes = num_scenes

        self.queries = torch.arange(num_queries, dtype=torch.long).unsqueeze(dim=1)
        self.scenes = torch.arange(num_scenes, dtype=torch.long)


    def __getitem__(self, index):
        q_t_grd = torch.from_numpy(self.q_t_graph[index].toarray()).squeeze()
        return index, q_t_grd


    def __len__(self):
        return self.q_t_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_queries, self.num_scenes, self.num_tools = self.get_data_size()

        s_t_pairs, s_t_graph = self.get_st()
        q_t_pairs_train, q_t_graph_train = self.get_qt("train")
        q_t_pairs_val, q_t_graph_val = self.get_qt("tune")
        q_t_pairs_test, q_t_graph_test = self.get_qt("test")

        q_s_pairs_train, q_s_graph_train = self.get_qs("train")
        q_s_pairs_val, q_s_graph_val = self.get_qs("tune")
        q_s_pairs_test, q_s_graph_test = self.get_qs("test")
        q_s_for_neg_sample, s_s_for_neg_sample = None, None

        self.scene_train_data = sceneTrainDataset(conf, s_t_pairs, q_s_pairs_train, q_t_graph_train, q_s_graph_train, self.num_tools, q_s_for_neg_sample, s_s_for_neg_sample, conf["neg_num"])
        self.scene_val_data = sceneTestDataset(q_s_pairs_val, q_t_graph_val, self.num_queries, self.num_scenes)
        self.scene_test_data = sceneTestDataset(q_s_pairs_test, q_t_graph_test, self.num_queries, self.num_scenes)

        self.graphs = [q_s_graph_train, q_t_graph_train, s_t_graph]

        self.train_loader = DataLoader(self.scene_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.scene_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.scene_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_aux_graph(self, q_t_graph, s_t_graph, conf):
        q_s_from_i = q_t_graph @ s_t_graph.T
        q_s_from_i = q_s_from_i.todense()
        bn1_window = [int(i*self.num_scenes) for i in conf['hard_window']]
        q_s_for_neg_sample = np.argsort(q_s_from_i, axis=1)[:, bn1_window[0]:bn1_window[1]]

        s_s_from_i = s_t_graph @ s_t_graph.T
        s_s_from_i = s_s_from_i.todense()
        bn2_window = [int(i*self.num_scenes) for i in conf['hard_window']]
        s_s_for_neg_sample = np.argsort(s_s_from_i, axis=1)[:, bn2_window[0]:bn2_window[1]]

        return q_s_for_neg_sample, s_s_for_neg_sample


    def get_st(self):
        with open(os.path.join(self.path, self.name, 'scene_tool.txt'), 'r') as f:
            s_t_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(s_t_pairs, dtype=np.int32)
        values = np.ones(len(s_t_pairs), dtype=np.float32)
        s_t_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_scenes, self.num_tools)).tocsr()

        print_statistics(s_t_graph, 'S-T statistics')

        return s_t_pairs, s_t_graph


    def get_qt(self, task):
        with open(os.path.join(self.path, self.name, 'query_tool_{}.txt'.format(task)), 'r') as f:
            q_t_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(q_t_pairs, dtype=np.int32)
        values = np.ones(len(q_t_pairs), dtype=np.float32)
        q_t_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_queries, self.num_tools)).tocsr()

        print_statistics(q_t_graph, 'Q-T statistics')

        return q_t_pairs, q_t_graph


    def get_qs(self, task):
        with open(os.path.join(self.path, self.name, 'query_scene_{}.txt'.format(task)), 'r') as f:
            q_s_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(q_s_pairs, dtype=np.int32)
        values = np.ones(len(q_s_pairs), dtype=np.float32)
        q_s_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_queries, self.num_scenes)).tocsr()

        print_statistics(q_s_graph, "Q-S statistics in %s" %(task))

        return q_s_pairs, q_s_graph