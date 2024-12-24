import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import binary_search
from utils import print_hms
import time
import numpy as np
from tqdm import tqdm
import pickle
import random
"""
Look up table은 시간에 따라 다르지 않고 공유하는 것으로 하자

"""
random_seed = 1024
random.seed(random_seed)
torch.manual_seed(random_seed)
class GCRNN(nn.Module):
    def __init__(self, user_num, comp_num, rel_num, emb_dim, user_id_max, cuda):
        super(GCRNN, self).__init__()
        self.device0 = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self.device2 = torch.device(cuda if torch.cuda.is_available() else "cpu")
        # self.device2 = torch.device("cpu")
        # self.device0 = torch.device("cpu")
        # self.device1 = torch.device("cpu")
        print("Utilizing", self.device0)
        self.user_num = user_num
        self.comp_num = comp_num
        self.entity_num = user_num + comp_num + 2
        # self.user_embedding_layer = nn.Embedding(self.user_num, emb_dim, sparse = False).to(self.device0) # RNN의 h0
        self.ent_embedding_layer = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0) # GCN의 초기값
        self.c0_embedding_layer_u = nn.Embedding(self.entity_num, emb_dim, sparse = False).to(self.device0) # for cell state in LSTM_GCN
        # self.c0_embedding_layer_c = nn.Embedding(self.user_num, emb_dim, sparse = False).to(self.device0) # for cell state in LSTM_comp
        # self.c0_embedding_layer_j = nn.Embedding(self.user_num, emb_dim, sparse = False).to(self.device0) # for cell state in LSTM_job
        self.rel_embedding_layer = nn.Embedding(rel_num, emb_dim, sparse = False).to(self.device0)
        self.rel_num = rel_num
        num_layers = 1

        # self.user_RNN = nn.RNNCell(emb_dim, emb_dim, bias = True).to(self.device0) # input dim, hn dim
        # self.comp_RNN = nn.RNNCell(emb_dim, emb_dim).to(self.device0) # input dim, hn dim
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0) # input dim, hn dim
        # self.comp_RNN_G = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0)
        # self.comp_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0) # input dim, hn dim
        # self.job_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device0) # input dim, hn dim
        # self.user_RNN = nn.GRUCell(emb_dim, emb_dim).to(self.device0) # input dim, hn dim
        # self.comp_RNN = nn.GRUCell(emb_dim, emb_dim).to(self.device0) # input dim, hn dim
        # h0는 ent_embedding_layer를 사용

        # self.comp_MLP = nn.Linear(emb_dim, comp_num).to(self.device0)
        # self.job_MLP = nn.Linear(emb_dim, rel_num//2).to(self.device0)

        # self.criterion = nn.CrossEntropyLoss()

        self.user_id_max = user_id_max

        # nn.init.xavier_normal_(self.ent_embedding_layer.weight.data)
        # nn.init.xavier_normal_(self.ent_embedding_layer_u.weight.data)
        # nn.init.xavier_normal_(self.ent_embedding_layer_c.weight.data)
        # nn.init.xavier_normal_(self.ent_embedding_layer_j.weight.data)
        # nn.init.xavier_normal_(self.rel_embedding_layer.weight.data)
        # nn.init.xavier_normal_(self.comp_MLP.weight.data)
        # nn.init.xavier_normal_(self.job_MLP.weight.data)
        
        # nn.init.xavier_normal_(self.comp_RNN.weight_ih)
        # nn.init.xavier_normal_(self.comp_RNN.weight_hh)
        # nn.init.xavier_normal_(self.user_RNN.weight_ih)
        # nn.init.xavier_normal_(self.user_RNN.weight_hh)
        # nn.init.xavier_normal_(self.comp_RNN.bias_ih)
        # nn.init.xavier_normal_(self.comp_RNN.bias_hh)
        # nn.init.xavier_normal_(self.user_RNN.bias_ih)
        # nn.init.xavier_normal_(self.user_RNN.bias_hh)
        print("With 1:1 Self Loop")
        # print("Sum reduce with degree norm")
        print("Mean reduce")
        print("1-hop")
        print("LSTM")
        # print("RNN")
        # print("GRU")
        print("Seq_enc inside GCN")

    def forward(self, user_batch, comp_batch, job_batch, start_batch, g, splitted_g, history_length, remove_list):
        seed_list = []
        seed_entid = []
        train_t = []
        # comp_labels = []
        # job_labels = []
        comp_target_0 = []
        comp_target_1 = []
        for comp_list in comp_batch:
            comp_target_0.append(torch.tensor([comp_list[0]]))
            comp_target_1.append(comp_list[1:])
        comp_target_0.extend(comp_target_1)

        job_target_0 = []
        job_target_1 = []
        for job_list in job_batch:
            job_target_0.append(torch.tensor([job_list[0]]))
            job_target_1.append(job_list[1:])
        job_target_0.extend(job_target_1)

        for time_list, user, comp_list in zip(start_batch, user_batch, comp_batch):
            # comp_list = (comp_list + self.user_num).tolist()
            for time in time_list[1:]:
                train_t.append(time)
                seed_entid.append(user)
            # for time, comp in zip(time_list[1:], comp_list[:-1]):
                # train_t.append(time)
                # seed_entid.append(comp)
        latest_train_time = max(train_t)
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user, comp_list in zip(start_batch, user_batch, comp_batch):
            # comp_list = (comp_list + self.user_num).tolist()
            for time in time_list[1:]:
                seed_list[time].add(user)
            # for time, comp in zip(time_list[1:], comp_list[:-1]): # 사실 이 for문은 이 윗 for문과 따로 만들 필요는 없다.
            #     seed_list[time].add(comp)
        ent_embs = self.seq_GCRNN_batch(g, splitted_g, latest_train_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), sorted = True, return_inverse = True)
        user_emb_0 = self.ent_embedding_layer(torch.tensor(user_batch).to(self.device0))

        # user_embs = ent_embs[index_for_ent_emb[:len(index_for_ent_emb)//2]]
        # comp_embs = ent_embs[index_for_ent_emb[len(index_for_ent_emb)//2:]]
        user_embs = ent_embs[index_for_ent_emb]
        u_time_embs = torch.cat([user_emb_0, user_embs]) # (N, emb_dim)

        target_c_embs = self.ent_embedding_layer(torch.cat(comp_target_0).to(self.device0) + self.user_id_max + 1) # (N, emb_dim)
        all_c_embs = self.ent_embedding_layer(torch.tensor(list(range(self.comp_num))).to(self.device0) + self.user_id_max + 1) # (comp_num, emb_dim)
        pos_score_comp = torch.sum(u_time_embs * target_c_embs, 1).unsqueeze(1) # (batch, 1)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1,0)) # (N, comp_num)
        comp_loss_procedure = pos_score_comp - torch.logsumexp(all_score_comp, 1).unsqueeze(1)
        comp_NLL_loss = -torch.sum(comp_loss_procedure)

        target_j_embs = self.rel_embedding_layer(torch.cat(job_target_0).to(self.device0)) # (N, emb_dim)
        all_j_embs = self.rel_embedding_layer(torch.tensor(list(range(self.rel_num//2))).to(self.device0)) # (job_num, emb_dim)
        pos_score_job = torch.sum(u_time_embs * target_j_embs, 1).unsqueeze(1) # (batch, 1)
        all_score_job = torch.matmul(u_time_embs, all_j_embs.transpose(1,0)) # (N, job_num)
        job_loss_procedure = pos_score_job - torch.logsumexp(all_score_job, 1).unsqueeze(1)
        job_NLL_loss = -torch.sum(job_loss_procedure)

        return comp_NLL_loss, job_NLL_loss

    def inference(self, comp_batch, job_batch, user_batch, test_time, g, splitted_g, history_length, remove_list):
        seed_list = []
        seed_entid = []
        test_t = []
        for i in range(test_time+1):
            seed_list.append(set())
        for user in user_batch:
            test_t.append(test_time)
            seed_entid.append(user)
            seed_list[test_time].add(user)
        # for comp_list in comp_batch:
        #     comp_list = (comp_list + self.user_num).tolist()
        #     latest_comp = comp_list[-1]
        #     test_t.append(test_time)
        #     seed_entid.append(latest_comp)
        #     seed_list[test_time].add(latest_comp)
        ent_embs = self.seq_GCRNN_batch(g, splitted_g, test_time, seed_list, history_length, remove_list)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * test_time + torch.tensor(test_t), sorted = True, return_inverse = True)
        u_time_embs = ent_embs[index_for_ent_emb]

        all_c_embs = self.ent_embedding_layer(torch.tensor(list(range(self.comp_num))).to(self.device0) + self.user_id_max + 1) # (comp_num, emb_dim)
        all_score_comp = torch.matmul(u_time_embs, all_c_embs.transpose(1,0)) # (N, comp_num)

        all_j_embs = self.rel_embedding_layer(torch.tensor(list(range(self.rel_num//2))).to(self.device0)) # (job_num, emb_dim)
        all_score_job = torch.matmul(u_time_embs, all_j_embs.transpose(1,0)) # (N, job_num)

        return all_score_comp, all_score_job
    
    # def msg_GCN(self,edges):  # out degree
    #     return {'m' : (edges.src['node_emb'] * self.rel_embedding[edges.data['relation_idx'].type(torch.LongTensor)]) / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    # def reduce_GCN(self,nodes): # in degree
    #     return {'node_emb2': nodes.mailbox['m'].sum(1)}

    def msg_GCN(self,edges):  # out degree
        return {'m' : edges.src['node_emb'] * self.rel_embedding[edges.data['relation_idx'].type(torch.LongTensor)]}
    # / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])

    def reduce_GCN(self,nodes): # in degree
        return {'node_emb2': nodes.mailbox['m'].mean(1)}
    # + nodes.data['node_emb'] * (len(nodes.mailbox['m']) / nodes.data['in_degree'])

    def update_node(self,nodes):
        return {'node_emb': nodes.data['node_emb'] + nodes.data['node_emb2']}
        
    def seq_GCRNN_batch(self, g, splitted_g, latest_train_time, seed_list, history_length, remove_list):
        """
        Input: (seed) 얻고싶은 entity와 그 시간 (entity, timestamp) * entity는 user와 company가 섞여있음
        Output: seed들의 임베딩

        주의점: 우리가 얻고 싶은 임베딩은 GCN->RNN->GCN->RNN의 임베딩이다.
        즉, 3번쨰 시점의 임베딩을 얻기 위해서는 1, 2 번째 시점의 임베딩도 필요하다.
        최적화를 위해서는, 같은 entity의 다른시점들이 한번에 들어오도록 하는 것이 좋지만 random학습을 위해 그럴 수는 없다.
        """

        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        gcn_1hopneighbor_per_time = []
        gcn_seed_2hopedge_per_time = []
        a2 = time.time()
        # print("Sampling neighbors...", end = " ")
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num + self.comp_num)
        # print("History length:", history_length)
        for i in range(latest_train_time, -1, -1): # latest -> 0 미래부터 본다.
            # print(len(future_needed_nodes))
            check_lifetime[list(seed_list[i])] = history_length # seed_list: time별로 seed user가 들어있음

            # seed list에 들어있는 user들을 future needed nodes에 추가함(과거로 미래를 initialize하기 때문)
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            # 따라서 해당 seed들은 과거에도 계속 seed에 들어가게 되지만, 과거에 edge가 존재하는지 여부는 모름
            # 또한 history length가 full(100)인 현 상황에서는 초반(앞 시간대)구간의 경우 seed수가 계속 같을 수 있음
            # 또한, 데이터 특성상 과거로 갈 수록 edge가 적음

            # 1hop edges of seed at i
            hop1_u, hop1_v = splitted_g[i].in_edges(v = list(future_needed_nodes), form = 'uv') #u -> v 이다.
            # hop1_neighbors_at_i, _, seed_edges_at_i = splitted_g[i].in_edges(v = list(future_needed_nodes), form = 'all')
            # node는 그대로 가져와지지만, splitted에서 추출한 edge id는 g의 edge id와 다를 수 있다.
            # 따라서 'edge id'가 아니라 'node id 쌍'로 edge를 기록해야 한다.

            # sample한 user들(entity)들에 대해 in-edge들 찾는다.
            # hop 1 neighbors는 2layer를 위해 찾아둔것

            # check_lifetime[hop1_neighbors_at_i] = history_length
            # hop2_edges_at_i = splitted_g[i].in_edges(v = hop1_neighbors_at_i, form = 'eid')
            # 2번째 layer를 위한 edge

            gcn_seed_per_time.append(list(future_needed_nodes)) # Seed
            # gcn에 seed로 사용되는 entity들이다. 사실 edge를 사용하기는 하지만..
            # 미래에서부터 쌓아왔기 때문에(사용하는 history length가 100이라서ㅋ) 과거로 갈수록 꽤나 양이 커진다.
            
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v)) # Seed's Edge
            #gcn_1hopneighbor_per_time.append(hop1_neighbors_at_i) # Seed's Edge's source node
            #gcn_seed_2hopedge_per_time.append(hop2_edges_at_i) # Source node's edge
            check_lifetime[check_lifetime>0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - remove_list[i-1] - set(np.where(check_lifetime==0)[0]) # seed next
            except:
                pass
        a3 = time.time()
        self.rel_embedding = self.rel_embedding_layer(torch.tensor(range(self.rel_num)).to(self.device0))
        last_entity_index = self.entity_num-2-1 # 1은 number를 index로 바꿔주려고 한 것 같고 2는 쓰레기 노드때문인듯
        # g = g.to(self.device0)
        g.ndata['node_emb'] = self.ent_embedding_layer(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.tensor(range(g.number_of_nodes())).to(self.device0))
        entity_embs = []
        entity_index = []
        # register함수는 DGL 0.9이상에서는 없어졌다.
        g.register_message_func(self.msg_GCN)
        g.register_reduce_func(self.reduce_GCN)
        for i in range(latest_train_time+1): # 0 -> latest
            # g_now = splitted_g[i]
            inverse = latest_train_time -i
            # gcn_seed_per_time -> 미래부터 들어있다
            if len(gcn_seed_per_time[inverse]) > 0:
                # g.ndata['in_degree_sqrt'] = torch.sqrt(g_now.in_degrees().type(torch.FloatTensor)).unsqueeze(1).to(self.device0)
                # g.ndata['out_degree_sqrt'] = torch.sqrt(g_now.out_degrees().type(torch.FloatTensor)).unsqueeze(1).to(self.device0)
                changed = sorted(gcn_seed_per_time[inverse])
                thresh = binary_search(changed, self.user_id_max + 1)
                user_seed_ = changed[:thresh]
                # comp_seed_ = changed[thresh:]
                user_seed_ = changed
                user_prev_hn = g.ndata['node_emb'][user_seed_]#.to(self.device1)
                # comp_prev_hn = g.ndata['node_emb'][comp_seed_]#.to(self.device1)
                user_prev_cn = g.ndata['cx'][user_seed_]#.to(self.device1)
                # comp_prev_cn = g.ndata['cx'][comp_seed_]#.to(self.device1)
                # if len(gcn_1hopneighbor_per_time[inverse]) > 0:
                #     g.send_and_recv(edges = gcn_seed_2hopedge_per_time[inverse],
                #                     message_func = self.msg_GCN3,
                #                     reduce_func = self.reduce_GCN3,
                #                     apply_node_func = self.update_node)
                # g.apply_nodes(func=self.update_node2, v=changed)
                # 1-hop
                edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                # g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse].to(self.device0),
                #                     message_func = self.msg_GCN3,
                #                     reduce_func = self.reduce_GCN3)
                g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse])
                if edge_num > 0:
                    try:
                        g.ndata['node_emb'] = g.ndata['node_emb2'] + g.ndata['node_emb']
                        g.ndata.pop('node_emb2')
                    except:
                        pass
                user_input = g.ndata['node_emb'][user_seed_]
                # comp_input = g.ndata['node_emb'][comp_seed_]
                # user_hn = self.user_RNN(user_input, user_prev_hn) # 바뀐부분만 RNN해줘야 하는것이 아닌지? 이부분 체크
                # comp_hn, comp_cn = self.comp_RNN_G(comp_input, (comp_prev_hn, comp_prev_cn)) # 예전에 comp의 이직순서고려하는 RNN이랑 구분하려고 G붙인듯
                # g.ndata['node_emb'][user_seed_] = user_hn
                # g.ndata['cx'][user_seed_] = user_cn
                # g.ndata['node_emb'][comp_seed_] = comp_hn
                # g.ndata['cx'][comp_seed_] = comp_cn
                # g.apply_nodes(func = self.update_node, v = user_seed_)
                # user_hn = g.ndata['node_emb2'][user_seed_]#.to(self.device1)
                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))
                g.ndata['node_emb'][user_seed_] = user_hn
                g.ndata['cx'][user_seed_] = user_cn
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i
                entity_embs.append(seed_emb)
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))
                # g.ndata.pop('reduced')
                # user_hn = self.user_RNN(user_input, user_prev_hn)
                # comp_hn = self.comp_RNN(comp_input, comp_prev_hn) # RNN결과값을 GCN초기값으로 안쓰네?
                # if len(comp_seed_) > 0 :
                #     comp_input = g.ndata['node_emb'][comp_seed_]#.to(self.device1)
                #     comp_hn, comp_cn = self.comp_RNN(comp_input, (comp_prev_hn, comp_prev_cn))
                #     g.ndata['node_emb'][comp_seed_] = comp_hn
                #     g.ndata['cx'][comp_seed_] = comp_cn
                #     comp_changed_in_global = torch.tensor(comp_seed_) * latest_train_time + i
                #     entity_embs.append(comp_hn)
                #     entity_index.append(comp_changed_in_global.type(torch.FloatTensor))
        entity_embs = torch.cat(entity_embs).to(self.device1)
        entity_index = torch.cat(entity_index)
        a4 = time.time()
        return entity_embs[entity_index.argsort()]

