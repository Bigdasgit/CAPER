import torch

def binary_search(list_, key):
  low = 0
  high = len(list_)-1
  while high >= low:
    mid = (low+high)//2
    if key<list_[mid]:
      high = mid-1
    elif key == list_[mid]:
      return mid
    else:
      low = mid+1
  return low

def print_hms(time):
    if time / 3600 > 1:
        print("{:.1f}h".format(time / 3600), end =" ")
        time %= 3600
    if time / 60 > 1:
        print("{:.1f}m".format(time / 60), end =" ")
        time %= 60
    print("{:.1f}s".format(time))

def inner_product_score(a, b):
  return torch.sum(a * b, dim=-1) 
# elementwise 곱 

def hermitian_product_score(a, b):
  real_a, imaginary_a = torch.chunk(a, 2, dim=-1)
  real_b, imaginary_b = torch.chunk(b, 2, dim=-1)
  return torch.sum(real_a * real_b - imaginary_a * imaginary_b)

def MRR(ranks):
    sum_ = 0
    for rank in ranks:
        sum_+= 1/rank
    return sum_ / len(ranks)

def HitsK(ranks, k):
    sum_ = 0
    for rank in ranks:
        if rank <= k:
            sum_+=1
    return (sum_/len(ranks))*100

def eval_rank(ranks):
    return MRR(ranks), HitsK(ranks,1), HitsK(ranks,3), HitsK(ranks,5), HitsK(ranks,10)

def print_metrics(comp_ranks, job_ranks):
    mrr, h1, h3, h5, h10 = eval_rank(comp_ranks)
    mrr_, h1_, h3_, h5_, h10_ = eval_rank(job_ranks)
    print('comp_prediction: MRR',round(mrr,4),'Hits@1', round(h1,4),'Hits@3', round(h3,4),'Hits@5', round(h5,4),'Hits@10', round(h10,4))
    print('job_prediction: MRR',round(mrr_,4),'Hits@1', round(h1_,4),'Hits@3', round(h3_,4),'Hits@5', round(h5_,4),'Hits@10', round(h10_,4))
    print('average:         MRR',round((mrr+mrr_)/2,4),'Hits@1', round((h1+h1_)/2,4),'Hits@3', round((h3+h3_)/2,4),'Hits@5', round((h5+h5_)/2,4),'Hits@10', round((h10+h10_)/2,4))
    return round((mrr+mrr_)/2,4), round((h1+h1_)/2,4), round((h3+h3_)/2,4), round((h5+h5_)/2,4),round((h10+h10_)/2,4)

def stabilized_log_softmax(pos, neg):
    return pos - torch.logsumexp(torch.cat([neg,pos.unsqueeze(1)],1), 1)

def stabilized_NLL(positive, negative):
    return -(torch.sum(stabilized_log_softmax(positive, negative)))

def rank(list_, key):
    try: 
        return len(list_) - list_.index(key)
    except: 
        return len(list_) - binary_search(list_,key) + 1

