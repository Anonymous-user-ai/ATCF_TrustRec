#4000个用户，2328个项，raw_data共有41172个非零项，密度为0.442139%,
# ratio=0.2时，data共有32928个非零项，密度为0.353608247%，有337个无效用户

import numpy as np

# 创建字典作为全局变量
u_dict={}
i_dict={}

def load_data_from_rating():
    file_path = '../../dataset/rating.txt'
    data = np.zeros((4000, 2328))
    # 初始化用户个数、项目个数，for循环作累加避免溢出
    u_cnt=-1
    i_cnt=-1

    with open(file_path,'r')as f:
        for line in f:
            line=line.replace('/','')#去除数据集中无效符号，才能读取数据
            vals=[int(x) for x in line.split()]
            #以下建立用户项的评分矩阵
            if vals[1] in u_dict:
                u_index=u_dict.get(vals[1])
            else:
                u_cnt+=1
                if u_cnt>=4000:break
                u_dict[vals[1]]=u_cnt
                u_index=u_dict.get(vals[1])
            if vals[0] in i_dict:
                i_index = i_dict.get(vals[0])
            else:
                i_cnt += 1
                if i_cnt >= 2328: break
                i_dict[vals[0]] = i_cnt
                i_index = i_dict.get(vals[0])
            data[u_index][i_index] = vals[2]
    return data


def erase_data(data, ratio = 1.0, seed = 1):

    if ratio == 1.0:
        return data

    total_count = data.size
    erase_count = (int)(total_count * ratio)

    np.random.seed(seed)
    indices = np.random.choice(total_count, erase_count, replace=False)
    filtered_data = np.copy(data)
    filtered_data.ravel()[indices] = 0

    return filtered_data

def Sparsity_caculation(data):
    # 该函数用于计算data数据矩阵的稀疏程度
    exist_rating=data!=0
    exist_num=np.sum(exist_rating)
    return exist_num/(data.shape[0]*data.shape[1])

def Find_valid_users(raw_data,data):
    # 该函数去除data中所有对任何项目无打分的用户（由于分数被抹掉造成的无打分，无法计算平均值，所以要把raw_data和data中对应的用户行都删除）
    valid_flags = data > 0
    valid_users = np.sum(valid_flags, axis=1)
    data=data[valid_users > 0, :]
    raw_data=raw_data[valid_users>0,:]
    return  raw_data,data

raw_data=load_data_from_rating()
data=erase_data(raw_data,0.2)




