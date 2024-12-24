# -*- coding: utf-8 -*-
import torch
import pickle5 as pickle
import time
import random
from tqdm import tqdm
import argparse
from collections import defaultdict
import numpy as np
from time_split_batch import time_split_graph
from utils import print_metrics
import model
import dgl
# +
#pip install dgl==0.4.3

# +
# parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser(description="Career Prediction")
# parser.add_argument("--history_length", default=100, type=int, help="past length for RNNs")
# parser.add_argument("--device", default="cuda:0", type=str, help="Which device do you wanna use")
# parser.add_argument("--train_until", default=2015, type=int, help="")
# parser.add_argument("--emb_dim", default=150, type=int, help="")
# args = parser.parse_args()
# -


# train_until = args.train_until - 1968
# history_length = args.history_length
# device = args.device
train_until = 2015 - 1968
history_length = 100
device = 'cuda:0'
random_seed = 1024
random.seed(random_seed)
torch.manual_seed(random_seed)

print("Loading data...")
with open('/home/user/EJHyun/Career Prediction/long_term/train_data.pickle', 'rb') as f:
    User_cnt, Comp_cnt, Job_cnt, Time_cnt, train_job, train_comp, train_dur, train_start, app_times = pickle.load(f)['data']
with open('/home/user/EJHyun/Career Prediction/long_term/test_data.pickle', 'rb') as f:
    user_future_companies, user_future_jobs, test_user_seed = pickle.load(f)['data']

remove_list = []
for i in range(Time_cnt):
    remove_list.append(set())
for i in range(User_cnt + Comp_cnt):
    try:
        remove_list[app_times[i][0]-1].add(i)
    except:
        pass
for i in range(Time_cnt):
    remove_list[i] = list(remove_list[i])

print('Loading graph...')
Train_Graph, splitted_Train_Graph = time_split_graph(train_until)

user_id_max = User_cnt-1
u, train_comp = zip(*list(train_comp.items()))
u, train_job = zip(*list(train_job.items()))
u, train_start = zip(*list(train_start.items()))
u, train_dur = zip(*list(train_dur.items()))


train_comp_tensor = []
train_job_tensor = []
fact_train_start = []

for d, c, j, s in zip(train_dur, train_comp, train_job, train_start):
    tmp_train_comp = []
    tmp_train_job = []
    tmp_train_start = []
    for dd, cc, jj, ss in zip(d, c, j, s):
        for i in range(dd+1):
            if ss+i <= train_until:
                tmp_train_comp.append(cc)
                tmp_train_job.append(jj)
                tmp_train_start.append(ss+i)
    train_comp_tensor.append(torch.tensor(tmp_train_comp).type(torch.LongTensor))
    train_job_tensor.append(torch.tensor(tmp_train_job).type(torch.LongTensor))
    fact_train_start.append(tmp_train_start)


test_user_entid, _ = zip(*test_user_seed)
test_comp = []
test_job = []
test_user = set(test_user_entid)
test_user_entid2 = []
for user, seq1, seq2 in zip(u, train_comp_tensor, train_job_tensor):
    if user in test_user:
        test_comp.append(seq1)
        test_job.append(seq2)
        test_user_entid2.append(user)
label_comps_index = defaultdict(dict)
for user in test_user_entid2:
    tmp = list(user_future_companies[user].keys())
    for time__ in range(48, 53):
        label_comps_index[user][time__] = torch.tensor([])
    for time__ in tmp:
        label_comps_index[user][time__] = (torch.tensor(list(user_future_companies[user][time__]))- (user_id_max + 1)).type(torch.LongTensor)

past_companies = dict()
for user in test_user_entid2:
    past_companies[user] = torch.unique(Train_Graph.in_edges(user, form = 'uv')[0]) - (user_id_max + 1)

# label_jobs_index = dict()
# for user in test_user_entid2:
#     label_jobs_index[user] = list(user_future_jobs[user])
label_jobs_index = defaultdict(dict)
for user in test_user_entid2:
    tmp = list(user_future_jobs[user].keys())
    for time__ in range(48, 53):
        label_jobs_index[user][time__] = torch.tensor([])
    for time__ in tmp:
        label_jobs_index[user][time__] = torch.tensor(list(user_future_jobs[user][time__]))

# emb_dim = args.emb_dim
emb_dim = 150
num_epochs = 100
trainset_batch_size = 500
test_batch_size = 500
best_mrr = 0
best_epoch = 0
learning_rate = 0.01 # 0.001


model = model.GCRNN(User_cnt, Comp_cnt, Job_cnt*2, emb_dim, User_cnt-1, device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
print("Train start")
for epoch in range(num_epochs):
    print("-epoch: ", epoch,"/ 0 ~",num_epochs-1,"processing")
    model.train()
    prev_batch_cnt = 0
    batch_cnt = 0
    epoch_start = time.time()
    for batch_it in tqdm(range(len(train_comp) // trainset_batch_size if len(train_comp) % trainset_batch_size == 0 else len(train_comp) // trainset_batch_size +1)):
        prev_batch_cnt = batch_cnt
        batch_cnt += trainset_batch_size
        if batch_cnt > len(train_comp):
            batch_cnt = len(train_comp)
        batch_size = batch_cnt - prev_batch_cnt
        comp_loss, job_loss = model(u[prev_batch_cnt:batch_cnt], train_comp_tensor[prev_batch_cnt:batch_cnt], train_job_tensor[prev_batch_cnt:batch_cnt], fact_train_start[prev_batch_cnt:batch_cnt], Train_Graph, splitted_Train_Graph, history_length, remove_list)
        loss = comp_loss + job_loss
        loss.backward() # calculate gradient
        optimizer.step() # update parameter via calculated gradient
        optimizer.zero_grad() # initialize
    print("Test start")
    model.eval()
    k1 = 2
    k2 = 1
    #print(len(test_comp))
    A = train_until
    with torch.no_grad():
        # for m in range(1,6):
        for m in range(1,2):
            company_ranks = []
            job_ranks = []
            prev_test_batch_cnt = 0
            test_batch_cnt = 0
            predicted_edges_u = []
            predicted_edges_v = []
            rel_idx = []
            for batch_it in tqdm(range(len(test_comp) // test_batch_size if len(test_comp) % test_batch_size == 0 else len(test_comp) // test_batch_size +1)):
                test_batch_cnt+=test_batch_size
                All_UC_score, All_UJ_score = model.inference(test_comp[prev_test_batch_cnt:test_batch_cnt], test_job[prev_test_batch_cnt:test_batch_cnt], test_user_entid2[prev_test_batch_cnt:test_batch_cnt], train_until + m - 1, Train_Graph, splitted_Train_Graph, history_length, remove_list)
                for user_id, UC_score, UJ_score in zip(test_user_entid2[prev_test_batch_cnt:test_batch_cnt], All_UC_score, All_UJ_score):
                    if len(label_comps_index[user_id][train_until + m]):
                        user_future_companies_index = label_comps_index[user_id][train_until + m]
                        Company_label_scores = UC_score[user_future_companies_index] # 정답 회사들의 점수들을 뽑는 과정이다.
                        for cls_ in Company_label_scores:
                            gap = UC_score - cls_
                            past_gap = gap[past_companies[user_id]]
                            company_ranks.append(len(gap[gap>0]) - len(past_gap[past_gap>0]) + 1)
                        Job_label_scores = UJ_score[label_jobs_index[user_id][train_until + m]]
                        for jls_ in Job_label_scores:
                            gap = UJ_score - jls_
                            job_ranks.append(len(gap[gap>0]) + 1)
                        comp_top_k1 = np.argsort(UC_score.to(torch.device("cpu")))[-k1:] + user_id_max
                        job_top_k2 = np.argsort(UJ_score.to(torch.device("cpu")))[-k2:]
                        for c in comp_top_k1:
                            predicted_edges_u.append(user_id)
                            predicted_edges_v.append(c)
                            predicted_edges_u.append(c)
                            predicted_edges_v.append(user_id)
                            rel_idx.append(job_top_k2[0])
                            rel_idx.append(job_top_k2[0])
                prev_test_batch_cnt = test_batch_cnt
            #print("m = ", m)
            mrr, h1, h3, h5, h10 = print_metrics(company_ranks, job_ranks)
            #=============================
            #train_until+=1
            remove_list.append([])
            predicted_graph = dgl.DGLGraph()
            predicted_graph.add_nodes(Train_Graph.number_of_nodes())
            predicted_graph.add_edges(predicted_edges_u, predicted_edges_v)
            predicted_graph.edata['relation_idx'] = torch.tensor(rel_idx)
            splitted_Train_Graph.append(predicted_graph)
    # if mrr > best_mrr:
    #     best_epoch = epoch
    #     best_mrr = mrr
    # print("Best MRR:", best_mrr, "at epoch", best_epoch)



