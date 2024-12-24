import dgl
from dgl.data.utils import load_graphs
import pickle
import torch
from tqdm import tqdm
import numpy as np

with open('/home/user/EJHyun/Career Prediction/long_term/quad.pickle', 'rb') as f:
    data = pickle.load(f)
head_list_id, tail_list_id, relation_list_id, time_list_id, user_cnt, comp_cnt, job_cnt, time_cnt, app_times = data['data']
entity_vocab, relation_vocab, time_vocab = list(range(user_cnt+comp_cnt)), list(range(job_cnt)), list(range(time_cnt))
user_id_max = user_cnt-1
with open('/home/user/EJHyun/Career Prediction/long_term/neg_and_app.pickle', 'rb') as f:
    C_app_dict, J_app_dict, uc_dict, uj_dict, jc_dict, cj_dict, app_times = pickle.load(f)['data']

def time_split_graph(train_until):
    Total_Graph = load_graphs('/home/user/EJHyun/Career Prediction/long_term/TKG.bin')[0][0]
    Time_span = len(time_vocab)
    time_mask = [time_ < train_until+1 for time_ in time_list_id]
    total_edge_index = list(range(len(time_list_id)*2))
    train_edge_index = torch.tensor(total_edge_index)[time_mask+time_mask]
    Train_Graph_byedge = Total_Graph.edge_subgraph(train_edge_index, preserve_nodes = True)
    Train_Graph_byedge.copy_from_parent()
    splitted = []
    for i in range(train_until+1):
        graph_at_i = Total_Graph.edge_subgraph(np.where(Total_Graph.edata['time_idx'] == i)[0], preserve_nodes = True)
        graph_at_i.copy_from_parent()
        splitted.append(graph_at_i)
    return Train_Graph_byedge, splitted

def time_split(train_until, history_length, Train_Graph):
    print("# of Nodes: ", Train_Graph.number_of_nodes())
    print("# of Edges: ", Train_Graph.number_of_edges())
    print("Masking...")
    train_mask = [time_ < train_until+1 for time_ in time_list_id]
    test_mask = [time_ > train_until for time_ in time_list_id]
    print("train data to list...")
    train_head = torch.tensor(head_list_id)[train_mask].tolist()
    train_tail = torch.tensor(tail_list_id)[train_mask].tolist()
    train_relation = torch.tensor(relation_list_id)[train_mask].tolist()
    train_time = torch.tensor(time_list_id)[train_mask].tolist()
    print("test data to list...")
    test_head = torch.tensor(head_list_id)[test_mask].tolist()
    test_tail = torch.tensor(tail_list_id)[test_mask].tolist()
    test_relation = torch.tensor(relation_list_id)[test_mask].tolist()
    test_time = torch.tensor(time_list_id)[test_mask].tolist()
    print("Getting quad_ids...")
    train_quadruples_ids = list(map(lambda x: (x[0], x[1], x[2], x[3]), zip(train_head, train_relation, train_tail, train_time)))
    reciprocal_train_quadruples_ids = list(map(lambda x: (x[2], x[1] + len(relation_vocab), x[0], x[3]), zip(train_head, train_relation, train_tail, train_time)))
    train_quadruples_ids.extend(reciprocal_train_quadruples_ids)
    # test_quadruples는 필요 없다.
    print("Listing test seeds...")
    test_user_seed = set()
    test_comp_seed = list(map(lambda x: (x, train_until + 1), list(set(tail_list_id)))) # Company는 user랑 다르게 다 뽑아야함
    print("Making Labels...")
    user_future_companies = dict()
    user_future_jobs = dict()
    print(len(test_head))
    train_head_set = set(train_head)
    for user, job, company in tqdm(zip(test_head, test_relation, test_tail)):
        if user in train_head_set:
            try:
                user_future_companies[user].add(company)
            except:
                user_future_companies[user] = set()
            try:
                user_future_jobs[user].add(job)
            except:
                user_future_jobs[user] = set()
            test_user_seed.add((user, train_until+1))
    return train_quadruples_ids, user_future_companies, user_future_jobs, user_id_max, entity_vocab, relation_vocab, time_vocab, list(test_user_seed), test_comp_seed, user_cnt, comp_cnt, job_cnt, time_cnt, app_times




