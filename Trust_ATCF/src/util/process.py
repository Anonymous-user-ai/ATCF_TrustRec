import numpy as np
import json
import random

# 为每一个用户和项设置特定的编号
u_dict = {}
i_dict = {}

def load_data_from_rating():
    with open('../../dataset/rating.txt', 'r') as f:
        # 初始化
        rating_matrix = np.zeros((4000, 2328))
        u_num = -1
        i_num = -1
        for line in f:
            line = line.replace('/', '')
            val = [int(x) for x in line.split()]
            if u_dict.get(val[1]) is None:
                u_num+=1
                if u_num >= 4000: break
                u_dict[val[1]] = u_num

            if i_dict.get(val[0]) is None:
                i_num += 1
                if i_num >= 2328: break
                i_dict[val[0]] = i_num

            idx1 = u_dict.get(val[1])
            idx2 = i_dict.get(val[0])
            rating_matrix[idx1][idx2] = val[2]

    return rating_matrix

def erase_data(data, ratio = 1.0, seed = 1):
    if ratio == 1.0:
        return data

    total = data.size
    erase_count = int(total * ratio)
    np.random.seed(seed)
    indices = np.random.choice(total, erase_count, replace=False)
    filtered_data = np.copy(data)
    filtered_data.ravel()[indices] = 0
    # c = np.sum(np.sum(filtered_data,axis=1)>0)
    # print(c)
    return filtered_data

def load_data_from_trust():
    with open('../../dataset/user_rating.txt') as f:
        trust_matrix = np.zeros((4000, 4000))
        for line in f:
            line = line.replace('/','')
            val = [int(x) for x in line.split()]
            if u_dict.get(val[0]) is not None and u_dict.get(val[1]) is not None:
                idx1 = u_dict.get(val[0])
                idx2 = u_dict.get(val[1])
                trust_matrix[idx1][idx2] = val[2]
    return trust_matrix

def store(ratio):
    rating_matrix = load_data_from_rating()
    filtered_data = erase_data(rating_matrix, ratio)
    trust_matrix = load_data_from_trust()
    # 去除不信任因素，去除自己对自己的信任
    trust_matrix[trust_matrix==-1] = 0
    for i in range(trust_matrix.shape[0]):
        for j in range(trust_matrix.shape[1]):
            if i == j and trust_matrix[i][j]>0:
                trust_matrix[i][j] = 0
    print(np.sum(rating_matrix))
    print(np.sum(filtered_data))
    print(np.sum(trust_matrix))
    with open('../../dataset/preprocessed_data_for_%.1fratio.json'%ratio,'w') as f:
        data = {}
        data['raw_matrix'] = rating_matrix.tolist()
        data['rating_matrix'] = filtered_data.tolist()
        data['trust_matrix'] = trust_matrix.tolist()
        json.dump(data, f)


if __name__ == '__main__':
    """
    ratio = 0.1 时，rating-matrix.shape = (3839,2328) 
                    trust-matrix.shape = (3839,3839)
    """
    store(0.2)




