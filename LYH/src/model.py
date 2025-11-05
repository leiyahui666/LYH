"""
@Time: 2022/9/3 17:52
@Author: hezf
@desc:
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GreyRelationalAnalysis(nn.Module):
    """
    灰关联分析模块，用于计算节点之间的均衡接近度
    """
    def __init__(self, zeta=0.5):
        """
        Args:
            zeta: 分辨系数，通常取0.5
        """
        super(GreyRelationalAnalysis, self).__init__()
        self.zeta = zeta
    
    def compute_grey_relational_coefficient(self, x0, xi):
        """
        计算灰关联系数 γ(x0(k),xi(k))
        Args:
            x0: 参考序列 (N, dim)
            xi: 比较序列 (N, M, dim)
        Returns:
            gamma: 灰关联系数 (N, M, dim)
        """
        # 计算绝对差值
        diff = torch.abs(x0.unsqueeze(1) - xi)  # (N, M, dim)
        
        # 计算 min_i min_k |x0(k) - xi(k)|
        min_diff = torch.min(diff.view(-1))
        
        # 计算 max_i max_k |x0(k) - xi(k)|
        max_diff = torch.max(diff.view(-1))
        
        # 计算灰关联系数
        numerator = min_diff + self.zeta * max_diff
        denominator = diff + self.zeta * max_diff
        gamma = numerator / (denominator + 1e-8)  # 添加小值避免除零
        
        return gamma
    
    def compute_grey_relational_degree(self, gamma):
        """
        计算灰关联度 γ(X0,Xi)
        Args:
            gamma: 灰关联系数 (N, M, dim)
        Returns:
            grey_degree: 灰关联度 (N, M)
        """
        # 对每个维度求平均
        grey_degree = torch.mean(gamma, dim=-1)
        return grey_degree
    
    def compute_equilibrium_proximity(self, gamma):
        """
        计算均衡接近度 B(X0,Xi)
        Args:
            gamma: 灰关联系数 (N, M, dim)
        Returns:
            equilibrium_proximity: 均衡接近度 (N, M)
        """
        # 计算灰关联度
        grey_degree = self.compute_grey_relational_degree(gamma)  # (N, M)
        
        # 计算灰关联密度 pi(k)
        gamma_sum = torch.sum(gamma, dim=-1, keepdim=True) + 1e-8  # (N, M, 1)
        pi = gamma / gamma_sum  # (N, M, dim)
        
        # 计算均衡度 B(Ri)
        # B(Ri) = -Σ pi(k)·ln(pi(k)) / ln(n)
        n = gamma.shape[-1]
        ln_pi = torch.log(pi + 1e-8)
        entropy = -torch.sum(pi * ln_pi, dim=-1)  # (N, M)
        equilibrium = entropy / (np.log(n) + 1e-8)
        
        # 计算均衡接近度 B(X0,Xi) = γ(X0,Xi)·B(Ri)
        equilibrium_proximity = grey_degree * equilibrium  # (N, M)
        
        return equilibrium_proximity
    
    def forward(self, hi, hj):
        """
        计算节点i和节点j之间的均衡接近度
        Args:
            hi: 节点i的特征 (N, dim)
            hj: 节点j的特征 (N, M, dim) 或 (N, N, dim)
        Returns:
            rho: 均衡接近度 (N, M) 或 (N, N)
        """
        gamma = self.compute_grey_relational_coefficient(hi, hj)
        rho = self.compute_equilibrium_proximity(gamma)
        return rho


class GraphAttentionLayer(nn.Module):
    """
    reference from https://github.com/Diego999/pyGAT
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    增强版：引入灰关联分析来改进注意力计算
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation_types: int = None, relation_dim: int = None):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.relation_types = relation_types
        self.relation_dim = relation_dim
        self.relation_embedding = None if self.relation_types is None else nn.Embedding(self.relation_types, self.relation_dim)
        self.W = nn.Linear(in_features=in_features, out_features=out_features)
        self.a = nn.Linear(2*out_features+(0 if self.relation_types is None else self.relation_dim), 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # 添加灰关联分析模块
        self.gra = GreyRelationalAnalysis(zeta=0.5)

    def forward(self, h, adj):
        Wh = self.W(h)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        # 计算灰色均衡接近度 ρhi,hj
        N = h.shape[0]
        h_repeated = h.unsqueeze(0).repeat(N, 1, 1)  # (N, N, in_features)
        rho = self.gra(h, h_repeated)  # (N, N) 灰色均衡接近度
        
        # 准备注意力机制输入
        a_input = self._prepare_attentional_mechanism_input(Wh)
        if self.relation_embedding is not None:
            a_input = torch.cat([a_input, self.relation_embedding(adj)], dim=-1)
        
        # 将灰色均衡接近度与注意力输入相乘（按照公式8）
        # e_ij = σ(a^T · W · ρ_{hi,hj} [hi||hj])
        rho_expanded = rho.unsqueeze(-1)  # (N, N, 1)
        a_input = a_input * rho_expanded  # 引入灰关联权重
        
        e = self.leakyrelu(self.a(a_input).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        if self.dropout > 0:
            attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, out_feature, dropout, alpha, nheads, relation_types: int = None, relation_dim: int = None):
        """
        nfeat：the dim of inputs
        nhid：the dim of hiddden state
        out_feature：the dim of outputs
        reference from https://github.com/Diego999/pyGAT
        Dense version of GAT.
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.relation_types = relation_types
        self.relation_dim = relation_dim
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True,
                                               relation_types=self.relation_types, relation_dim=self.relation_dim) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, out_feature, dropout=dropout, alpha=alpha, concat=False,
                                           relation_types=self.relation_types, relation_dim=self.relation_dim)

    def forward(self, x, adj):
        inputs = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + inputs


# class GraphAttentionWithRelationLayer(nn.Module):
#     """
#     带关系的图注意力层
#     增强版：引入灰关联分析来改进注意力计算
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionWithRelationLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#         self.W = nn.Linear(in_features=in_features, out_features=out_features)
#         self.w1 = nn.Linear(out_features, out_features)
#         self.a = nn.Linear(2 * out_features+2*in_features, 1)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#         # 添加灰关联分析模块
#         self.gra = GreyRelationalAnalysis(zeta=0.5)
#
#     def forward(self, h, e_h, adj, dialogue_embedding):
#         # dialogue_embedding.shape: (N, N, hidden_size)  adj.shape: (N, N)
#         Wh = self.W(h)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#
#         # 计算灰色均衡接近度 ρhi,hj
#         N = h.shape[0]
#         h_repeated = h.unsqueeze(0).repeat(N, 1, 1)  # (N, N, in_features)
#         rho = self.gra(h, h_repeated)  # (N, N) 灰色均衡接近度
#
#         # 准备注意力机制输入
#         a_input = self._prepare_attentional_mechanism_input(Wh)
#         a_input = torch.cat([a_input, e_h, dialogue_embedding], dim=-1)
#
#         # 将灰色均衡接近度与注意力输入相乘（按照公式8）
#         # e_ij = σ(a^T · W · ρ_{hi,hj} [hi||hj])
#         rho_expanded = rho.unsqueeze(-1)  # (N, N, 1)
#         a_input = a_input * rho_expanded  # 引入灰关联权重
#
#         e = self.leakyrelu(self.a(a_input).squeeze(2))
#
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         if self.dropout > 0:
#             attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def _prepare_attentional_mechanism_input(self, Wh):
#         N = Wh.size()[0]  # number of nodes
#         Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
#         Wh_repeated_alternating = Wh.repeat(N, 1)
#         all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
#         return all_combinations_matrix.view(N, N, 2 * self.out_features)


# class GATWithRelation(nn.Module):
#     def __init__(self, nfeat, nhid, out_feature, dropout, alpha, nheads):
#         """
#         nfeat：the dim of inputs
#         nhid：the dim of hiddden state
#         out_feature：the dim of outputs
#         Dense version of GAT.
#         """
#         super(GATWithRelation, self).__init__()
#         self.dropout = dropout
#         self.dialog_w = nn.Linear(out_feature, nfeat)
#         self.attentions = [GraphAttentionWithRelationLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = GraphAttentionWithRelationLayer(nhid * nheads, out_feature, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, h, edge_h, adj, dialogue_embedding):
#         # h.shape: (N, hidden_size)
#         # edge_h.shape: (N, N, hidden_size) adj.shape: (N, N) dialogue_embedding.shape: (hidden_size)
#         dialogue_embedding = self.dialog_w(dialogue_embedding)
#         dialogue_embedding = dialogue_embedding.repeat(adj.shape[0]**2, 1).view(adj.shape[0], adj.shape[0], -1)
#         inputs = h
#         x = F.dropout(h, self.dropout, training=self.training)
#         x = torch.cat([att(x, edge_h, adj, dialogue_embedding) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, edge_h, adj, dialogue_embedding))
#         return x + inputs


class PositionEncoder(nn.Module):
    def __init__(self, hidden_size, n_pos=200):
        """
        :param hidden_size: dim of position embedding
        :param n_pos: max length of seq
        """
        super(PositionEncoder, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(hidden_size, n_pos))

    def _get_sinusoid_encoding_table(self, hidden_size, n_pos=200):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(j//2)/hidden_size) for j in range(hidden_size)]

        sinusoid_table = np.array([get_position_angle_vec(i) for i in range(n_pos)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table)

    def forward(self, x):
        # x: (dialogue_len, hidden_size)
        return x + self.pos_table[x.shape[0]].clone().detach()


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_size, 1)

    def forward(self, embeddings):
        # embeddings: (n, embedding_size)
        a = self.w(embeddings).softmax(dim=0).view(1, -1)
        embedding = torch.mm(a, embeddings).squeeze(0)
        return embedding


class PTMDialogueEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(PTMDialogueEncoder, self).__init__()
        self.ptm = kwargs['model']
        self.gat = GAT(nfeat=kwargs['hidden_size'],
                       nhid=kwargs['hidden_size'],
                       out_feature=kwargs['hidden_size'],
                       dropout=0.3,
                       alpha=0.3,
                       nheads=1,
                       relation_types=kwargs.get('relation_types', None),
                       relation_dim=kwargs.get('relation_dim', None))
        self.position_encoder = PositionEncoder(kwargs['hidden_size'])
        self.attention = Attention(kwargs['hidden_size'])
        self.linear = nn.Linear(kwargs['hidden_size'], kwargs['output_size'])

    def forward(self, input_ids, token_type_ids, attention_mask, adj=None, cls_ids=None):
        utter_embedding = self.ptm(input_ids, attention_mask, token_type_ids)[0]
        new_utter_embed = []
        for batch_id, cls_id in enumerate(cls_ids):
            utter_embed = torch.index_select(utter_embedding[batch_id], 0, cls_id.to(utter_embedding.device))
            utter_embed = self.position_encoder(utter_embed)
            utter_embed = self.gat(utter_embed, adj[batch_id].to(input_ids.device))
            new_utter_embed.append(self.attention(utter_embed))
        dialogue_embedding = torch.stack(new_utter_embed)
        dialogue_embedding = self.linear(dialogue_embedding)
        return dialogue_embedding


# class KGEncoder(nn.Module):
#     def __init__(self, **kwargs):
#         super(KGEncoder, self).__init__()
#         if 'entity_embedding' in kwargs and 'relation_embedding' in kwargs:
#             self.entity_embedding = nn.Embedding.from_pretrained(kwargs['entity_embedding'], freeze=False)
#             self.relation_embedding = nn.Embedding.from_pretrained(kwargs['relation_embedding'], freeze=False)
#             self.W_e = None
#             self.W_r = None
#         else:
#             self.entity_embedding = nn.Embedding(kwargs['entity_nums'], kwargs['dim'])
#             self.relation_embedding = nn.Embedding(kwargs['relation_nums'], kwargs['dim'])
#             nn.init.xavier_normal(self.entity_embedding.weight)
#             nn.init.xavier_normal(self.relation_embedding.weight)
#             self.W_e = None
#             self.W_r = None
#         self.gat = GATWithRelation(nfeat=kwargs['dim'],
#                                    nhid=kwargs['dim'],
#                                    out_feature=kwargs['output_size'],
#                                    dropout=0.1,
#                                    alpha=0.3,
#                                    nheads=1)

    # (self, h, edge_h, adj, dialogue_embedding):
    # def forward(self, dialogue_embedding):
    #     # 移除了基于疾病的知识图谱编码，仅返回对话嵌入
    #     return dialogue_embedding


class DDN(nn.Module):
    def __init__(self, **kwargs):
        super(DDN, self).__init__()
        self.dialogue_encoder = PTMDialogueEncoder(**kwargs)
        #self.kg_encoder = KGEncoder(**kwargs)
        #self.classifier = nn.Linear(2*kwargs['output_size'], kwargs['n_label'])
        self.classifier = nn.Linear(kwargs['output_size'], kwargs['n_label'])

    def forward(self, input_ids, token_type_ids, attention_mask, adj, cls_ids):
        dialogue_embed = self.dialogue_encoder(input_ids, token_type_ids, attention_mask, adj, cls_ids)
        emb = dialogue_embed
        logits = self.classifier(emb)
        logits = torch.sigmoid(logits)
        return logits
