#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import json
import pandas as pd
from sentence_transformers import SentenceTransformer,util
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def cal_multi_label_loss(pred, label):
    loss = -(label * torch.log(pred) + (1 - label) * torch.log(1 - pred))
    loss = torch.mean(loss)
    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class COLT(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.base_model_name = conf["base_model_name"]
        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_queries = conf["num_queries"]
        self.num_scenes = conf["num_scenes"]
        self.num_tools = conf["num_tools"]
        self.dataset = conf["dataset"]

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.get_scene_agg_graph_ori()
        self.init_emb()

        self.get_tool_level_graph_ori()
        self.get_scene_level_graph_ori()
        self.get_scene_agg_graph_ori()

        self.get_tool_level_graph()
        self.get_scene_level_graph()
        self.get_scene_agg_graph()

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]


    def init_md_dropouts(self):
        self.tool_level_dropout = nn.Dropout(self.conf["tool_level_ratio"], True)
        self.scene_level_dropout = nn.Dropout(self.conf["scene_level_ratio"], True)
        self.scene_agg_dropout = nn.Dropout(self.conf["scene_agg_ratio"], True)


    def init_emb(self):
        base_model = './PLMs/{}'.format(self.base_model_name)
        encode_model = SentenceTransformer(base_model)
        query_file_path = './datasets/{}/queries.jsonl'.format(self.dataset)
        tool_file_path = './datasets/{}/corpus.jsonl'.format(self.dataset)
        file_path = './datasets/{}/query_tool_test.txt'.format(self.dataset)
        unique_numbers = set()
        with open(file_path, 'r') as file:
            for line in file:
                number = int(line.split('\t')[0])
                unique_numbers.add(number)
        self.test_index = list(unique_numbers)
        self.test_index.sort()
        queries = []
        tools = []
        test_queries =[]
        with open(query_file_path, 'r', encoding='utf-8') as query_file,open(tool_file_path, 'r', encoding='utf-8') as tool_file:
            for line in query_file:
                query = json.loads(line)
                queries.append(query)
                if int(query['_id'])in self.test_index:
                    test_queries.append(query)
            for line in tool_file:
                tool = json.loads(line)
                tools.append(tool)
        sorted_queries = sorted(queries, key=lambda x: int(x['_id']))
        sorted_test_queries = sorted(test_queries, key=lambda x: int(x['_id']))
        query_texts = [query['text'] for query in sorted_queries]
        test_query_texts =[query['text'] for query in sorted_test_queries]
        self.test_queries_feature=nn.Parameter(encode_model.encode(test_query_texts, show_progress_bar=True, convert_to_tensor=True).clone().detach())
        self.queries_feature = nn.Parameter(encode_model.encode(query_texts, show_progress_bar=True, convert_to_tensor=True).clone().detach())
        tool_texts=[tool['text'] for tool in tools]
        self.tools_feature = nn.Parameter(encode_model.encode(tool_texts, show_progress_bar=True, convert_to_tensor=True).clone().detach())
        self.scenes_feature = torch.matmul(self.scene_agg_graph_ori, self.tools_feature)
        self.index_tensor = torch.tensor(self.test_index).unsqueeze(1).expand(-1, self.queries_feature.size(1)).to(self.device)


    def get_tool_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["tool_level_ratio"]

        tool_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = tool_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                tool_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.tool_level_graph = to_tensor(laplace_transform(tool_level_graph)).to(device)


    def get_tool_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        tool_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.tool_level_graph_ori = to_tensor(laplace_transform(tool_level_graph)).to(device)


    def get_scene_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["scene_level_ratio"]

        scene_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = scene_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                scene_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.scene_level_graph = to_tensor(laplace_transform(scene_level_graph)).to(device)


    def get_scene_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        scene_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.scene_level_graph_ori = to_tensor(laplace_transform(scene_level_graph)).to(device)


    def get_scene_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["scene_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        scene_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/scene_size.A.ravel()) @ bi_graph
        self.scene_agg_graph = to_tensor(bi_graph).to(device)


    def get_scene_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        scene_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/scene_size.A.ravel()) @ bi_graph
        self.scene_agg_graph_ori = to_tensor(bi_graph).to(device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: 
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def get_TL_scene_rep(self, TL_tools_feature, test):
        if test:
            TL_scenes_feature = torch.matmul(self.scene_agg_graph_ori, TL_tools_feature)
        else:
            TL_scenes_feature = torch.matmul(self.scene_agg_graph, TL_tools_feature)

        if self.conf["scene_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            TL_scenes_feature = self.scene_agg_dropout(TL_scenes_feature)

        return TL_scenes_feature


    def propagate(self, test=False):
        if test:
            with torch.no_grad():
                 self.queries_feature.scatter_(0,self.index_tensor, self.test_queries_feature)
        if test:
            TL_queries_feature, TL_tools_feature = self.one_propagate(self.tool_level_graph_ori, self.queries_feature, self.tools_feature, self.tool_level_dropout, test)
        else:
            TL_queries_feature, TL_tools_feature = self.one_propagate(self.tool_level_graph, self.queries_feature, self.tools_feature, self.tool_level_dropout, test)

        TL_scenes_feature = self.get_TL_scene_rep(TL_tools_feature, test)

        if test:
            SL_queries_feature, SL_scenes_feature = self.one_propagate(self.scene_level_graph_ori, self.queries_feature, self.scenes_feature, self.scene_level_dropout, test)
        else:
            SL_queries_feature, SL_scenes_feature = self.one_propagate(self.scene_level_graph, self.queries_feature, self.scenes_feature, self.scene_level_dropout, test)

        queries_feature = [TL_queries_feature, SL_queries_feature]
        scenes_feature = [TL_scenes_feature, SL_scenes_feature]
        tool_feature=[TL_tools_feature, TL_tools_feature]

        return queries_feature, scenes_feature,tool_feature


    def cal_c_loss(self, pos, aug):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]
        
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) 
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) 

        pos_score = torch.exp(pos_score / self.c_temp) 
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) 

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, queries_feature, scenes_feature, tools_feature, label):
        TL_queries_feature, SL_queries_feature = queries_feature
        
        TL_scenes_feature, SL_scenes_feature = scenes_feature
        TL_tools_feature, SL_tools_feature = tools_feature
        TL_queries_feature = torch.nn.functional.normalize(TL_queries_feature, p=2, dim=2)
        TL_tools_feature = torch.nn.functional.normalize(TL_tools_feature, p=2, dim=2)
        SL_queries_feature = torch.nn.functional.normalize(SL_queries_feature, p=2, dim=2)
        SL_tools_feature = torch.nn.functional.normalize(SL_tools_feature, p=2, dim=2)
        pred = F.cosine_similarity(TL_queries_feature, TL_tools_feature, dim=2)+F.cosine_similarity(SL_queries_feature, SL_tools_feature, dim=2)
        pred = torch.nn.functional.softmax(pred, dim=1)
        zero_label = torch.zeros_like(label)
        label = zero_label + label / (torch.sum(label, dim=1).unsqueeze(1).expand(label.shape))
        multi_label_loss = cal_multi_label_loss(pred, label)

        u_cross_view_cl = self.cal_c_loss(TL_queries_feature, SL_queries_feature)
        b_cross_view_cl = self.cal_c_loss(TL_scenes_feature, SL_scenes_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return multi_label_loss, c_loss


    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.get_tool_level_graph()
            self.get_scene_level_graph()
            self.get_scene_agg_graph()


        queries, scene, tools, label= batch
        queries_feature, scenes_feature, tools_feature = self.propagate()

        queries_embedding = [i[queries].expand(-1, tools.shape[1], -1) for i in queries_feature]
        scene_embedding = [i[scene] for i in scenes_feature]
        tools_embedding = [i[tools] for i in tools_feature]
        multi_label_loss, c_loss = self.cal_loss(queries_embedding, scene_embedding, tools_embedding, label)

        return multi_label_loss, c_loss


    def evaluate(self, propagate_result, queries):
        queries_feature, scenes_feature, tools_feature = propagate_result
        queries_feature_atom, queries_feature_non_atom = [i[queries] for i in queries_feature]
        tools_feature_atom, tools_feature_non_atom = tools_feature
        queries_feature=self.queries_feature[queries]
        scores = cos_sim(queries_feature_atom, tools_feature_atom)+cos_sim(queries_feature_non_atom, tools_feature_non_atom)
        return scores
