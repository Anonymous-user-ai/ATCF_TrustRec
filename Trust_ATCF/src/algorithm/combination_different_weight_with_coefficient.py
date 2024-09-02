"""
该文件计算了综合upcc、ipcc、trust三个方法对用户偏好的预测
且用True和False区分了weight值是否是平均
"""


import numpy as np
import json
import time
from src.algorithm.U_method_with_coefficient import upcc
from  src.algorithm.I_method_fill_with_coefficient import ipcc
from  src.algorithm.Trust_method_with_coefficient import trust


def generate_predicted_of_three_method(seed, k1=1/3,k2=1/3,k3=1/3, top = 5):
    with open('../../dataset/preprocessed_data_for_0.2ratio.json', 'r') as f:
        data_dict = json.load(f)
        raw_data = np.array(data_dict['raw_matrix'])
        data = np.array(data_dict['rating_matrix'])
        np.random.seed(seed)
        user_indices = np.random.choice(range(data.shape[0]), 500, replace=False)
        item_indices = np.random.choice(range(data.shape[1]), 300, replace=False)

        with open('../../output/datafile_itself/K_choice/more_upcc_predicted_matrix_top=15.json', 'r')as f:
            predicted = json.load(f)
            upcc_predicted = predicted['predicted matrix']
        with open('../../output/datafile_itself/K_choice/more_ipcc_predicted_matrix_top=15.json', 'r')as f:
            predicted = json.load(f)
            ipcc_predicted = predicted['predicted matrix']
        with open('../../output/datafile_itself/K_choice/more_trust_predicted_matrix_top=15.json', 'r')as f:
            predicted = json.load(f)
            trust_predicted = predicted['predicted matrix']


        user_weight =  np.zeros(data.shape)
        item_weight =  np.zeros(data.shape)
        trust_weight = np.zeros(data.shape)
        user_weight[:, :] = k1
        item_weight[:, :] = k2
        trust_weight[:,:] = k3
        print(user_weight, item_weight, trust_weight)

    return  upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
            item_weight, trust_weight, data, raw_data, user_indices


class combination_for_different_weight:
    def __init__(self, upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
                 item_weight, trust_weight, data, raw_data, indices):
        self.upcc_predicted = upcc_predicted
        self.ipcc_predicted = ipcc_predicted
        self.trust_predicted = trust_predicted
        self.user_weight = user_weight
        self.item_weight = item_weight
        self.trust_weight = trust_weight
        self.data = data
        self.raw_data = raw_data
        self.indices = indices

    def evaluate(self):
        # 初始化
        mae = 0
        rmse = 0
        num_of_success = 0
        num_of_predicted = 0
        num_of_failure = 0
        coverage_flag = [False for x in range(self.data.shape[1])]

        # 求所有用户的平均分数
        valid_flag = self.data > 0
        valid_flag_count = np.sum(valid_flag, axis=1)
        valid_flag_count[valid_flag_count == 0] = 1
        users_avg_value = np.sum(self.data, axis=1) / valid_flag_count
        # 若用户没有任何打分，则将平均数置为全局平均数
        users_avg_value[users_avg_value == 0] = np.sum(self.data) / np.sum(self.data > 0)
        users_avg_value.reshape(self.data.shape[0], 1)

        # 计算三个方法的综合预测结果
        user_rating = np.multiply(self.user_weight, self.upcc_predicted)
        item_rating = np.multiply(self.item_weight, self.ipcc_predicted)
        trust_rating = np.multiply(self.trust_weight, self.trust_predicted)


        # 标记三种方法分别是否是有效方法，即能预测出来则为有效方法
        user_rating_flag = user_rating > 0
        item_rating_flag = item_rating > 0
        trust_rating_flag = trust_rating > 0

        # 能预测出来的方法才可以用来预测
        fenmu = np.multiply(self.user_weight, user_rating_flag) + \
                np.multiply(self.item_weight, item_rating_flag) + \
                np.multiply(self.trust_weight, trust_rating_flag)
        fenmu[fenmu == 0] = 1
        rating = (user_rating + item_rating + trust_rating) / fenmu

        #对于预测的每一个用户，计算各个指标的值
        for idx in self.indices:
            predicted_value = rating[idx]
            predicted_columns = np.multiply(self.data[idx] == 0, self.raw_data[idx] > 0)
            failure_columns = np.multiply(predicted_value == 0, predicted_columns == True)
            success_columns = np.multiply(predicted_value>0 ,predicted_columns == True)

            # 无法预测的值取平均
            predicted_value = predicted_value[predicted_columns>0]
            predicted_value[predicted_value == 0] = users_avg_value[idx]
            # print('预测分数', predicted_value)

            # 迭代计算各个指标值
            data_for_reference = self.raw_data[idx, predicted_columns > 0]
            mae += np.sum(np.abs(predicted_value - data_for_reference))
            rmse += np.sum(np.square(predicted_value - data_for_reference))
            coverage_flag = coverage_flag[:] | success_columns[:]
            num_of_predicted += np.sum(predicted_columns)
            num_of_failure += np.sum(failure_columns)
            num_of_success += np.sum(success_columns)
        # print('失败个数',num_of_failure)
        # print('预测个数',num_of_predicted)
        # print('覆盖个数',np.sum(coverage_flag))
        # print('成功个数',np.sum(num_of_success))
        return mae / num_of_predicted, np.sqrt(rmse / num_of_predicted), num_of_failure / num_of_predicted, \
               np.sum(np.array(coverage_flag)) / self.data.shape[1]

def compare_with_different_itemK(seed=1):
    # 该方法用于调参，upcc、ipcc、trust中ipcc占多大比例合适
    # accuracy的行分别代表 mae rmse failureRate coverage指标,列代表0-1每一个k值，以0.1为步长
    accuracy = np.zeros((4,11))
    upcc_predicted, ipcc_predicted, trust_predicted, data, raw_data = load_predicted_data()
    for (idx, k) in enumerate( np.arange(0, 1.1, 0.1)):
        user_weight, item_weight, trust_weight, indices = \
            generate_predicted_of_three_method(seed=1, k1=(1-k)/2, k2=k, k3=(1-k)/2, flag=True, top=15)
        combination_algorithm = combination_for_different_weight(upcc_predicted, ipcc_predicted, trust_predicted,\
                                                                 user_weight, item_weight,trust_weight,  data, \
                                                                 raw_data, indices)

        mae,rmse,failureRate,coverage = combination_algorithm.evaluate()
        accuracy[0][idx]=mae
        accuracy[1][idx]=rmse
        accuracy[2][idx]=failureRate
        accuracy[3][idx]=coverage
    return accuracy

def compare_with_different_userK(seed = 1):
    accuracy = np.zeros((4, 7))
    upcc_predicted, ipcc_predicted, trust_predicted, data, raw_data = load_predicted_data()
    for (idx, k) in enumerate(np.arange(0, 0.7, 0.1)):
        user_weight, item_weight, trust_weight, indices = \
            generate_predicted_of_three_method(seed=1, k1=1/3, k2=2/3, k3=0, flag=True, top=15)
        combination_algorithm = combination_for_different_weight(upcc_predicted, ipcc_predicted, trust_predicted, \
                                                                 user_weight, item_weight, trust_weight, data, \
                                                                 raw_data, indices)

        mae, rmse, failureRate, coverage = combination_algorithm.evaluate()
        accuracy[0][idx] = mae
        accuracy[1][idx] = rmse
        accuracy[2][idx] = failureRate
        accuracy[3][idx] = coverage
    return accuracy

def compare_with_average_K(seed=2):
    flag = False
    upcc_predicted, ipcc_predicted, trust_predicted, user_weight, item_weight, trust_weight, \
    data, raw_data, indices_users = generate_predicted_of_three_method(seed, None, flag, top=5)
    combination_algorithm = combination_for_different_weight(upcc_predicted, ipcc_predicted, trust_predicted, \
                                                             user_weight, item_weight, trust_weight, data, \
                                                             raw_data, indices_users)
    mae, rmse, failureRate, coverage = combination_algorithm.evaluate()
    # print('mae= ', mae, 'rmse= ', rmse, 'failureRate= ', failureRate, 'coverage= ', coverage)
    return mae, rmse, failureRate, coverage

def compare_with_best_K(seed = 2):
    flag = True
    upcc_predicted, ipcc_predicted, trust_predicted, user_weight, item_weight, trust_weight, \
    data, raw_data, indices_users = generate_predicted_of_three_method(seed, 0.3, flag, top=5)
    combination_algorithm = combination_for_different_weight(upcc_predicted, ipcc_predicted, trust_predicted, \
                                                             user_weight, item_weight, trust_weight, data, \
                                                             raw_data, indices_users)
    mae, rmse, failureRate, coverage = combination_algorithm.evaluate()
    # print('mae= ', mae, 'rmse= ', rmse, 'failureRate= ', failureRate, 'coverage= ', coverage)
    return mae, rmse, failureRate, coverage




if __name__ == '__main__':
    begin = time.time()
    upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
    item_weight, trust_weight, data, raw_data, indices_users = generate_predicted_of_three_method(1,k1=0,k2=0,k3=1)
    algorithm = combination_for_different_weight(upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
            item_weight, trust_weight, data, raw_data, indices_users )
    mae, rmse, failureRate, coverage = algorithm.evaluate()
    print('mae= ', mae, 'rmse= ', rmse, 'failureRate= ', failureRate, 'coverage= ', coverage)
    cost = time.time()-begin
    print('cost=',cost)

        # accuracy = compare_with_different_itemK(1)
    # print(accuracy)
    # with open('../../output/datafile_itself/compare_with_different_itemK','w') as fp:
    #     json.dump({'accuracy':accuracy.tolist()},fp)
    # accuracy = compare_with_different_userK(1)
    # print(accuracy)
    # with open('../../output/datafile_itself/compare_with_different_userK','w') as fp:
    #     json.dump({'accuracy':accuracy.tolist()},fp)
    # compare_with_average_K()
    # compare_with_best_K()




