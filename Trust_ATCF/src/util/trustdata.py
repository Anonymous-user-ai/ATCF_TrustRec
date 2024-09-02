# 信任矩阵维度为3663*3663，其中有效的信任用户个数为3189
#共有109242个信任关系（去除了不信任关系），矩阵的密度为0.81417133%

import numpy as np
from src.util import dataset
import random

def load_data_from_trust():
    file_path='../../dataset/user_rating.txt'
    trust_matrix=np.zeros((4000,4000))
    # 获取dataset中得到的用户字典
    u_dict = dataset.u_dict

    with open(file_path, 'r')as f:
        for line in f:
            line = line.replace('/', '')
            vals = [int(x) for x in line.split()]
            # 以下建立用户和用户信任关系的矩阵
            if vals[0] in u_dict and vals[1] in u_dict:
                user1 = u_dict.get(vals[0])
                user2 = u_dict.get(vals[1])
                trust_matrix[user1][user2]=vals[2]
    trust_matrix[trust_matrix<0]=0
    return trust_matrix

def Find_valid_users(data,trust_matrix):
    # dataset中得到的无效用户不应在信任矩阵中出现，使用该算法把相同的无效用户在信任矩阵中也删除
    valid_flags = data > 0
    valid_users = np.sum(valid_flags, axis=1)
    trust_matrix=trust_matrix[valid_users>0,:]
    trust_matrix=trust_matrix[:,valid_users>0]
    return trust_matrix

raw_data=dataset.load_data_from_rating()
data=dataset.erase_data(raw_data,0.2)
TM=load_data_from_trust()
TM=Find_valid_users(data,TM)

