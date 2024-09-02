"""
该文件计算了综合upcc、ipcc、trust三个方法对用户偏好的预测
且用True和False区分了weight值是否是平均
"""


import numpy as np
import json
import time
from src.compare_method.U_method_fill import upcc
from  src.compare_method.I_method_fill import ipcc
from  src.compare_method.Trust_method import trust

def generate_predicted_of_three_method(seed, top, k1=1/3, k2=1/3, k3=1/3):
    # 加载数据
    with open('../../dataset/preprocessed_data_for_0.2ratio.json', 'r') as f:
        data_dict = json.load(f)
        raw_data = np.array(data_dict['raw_matrix'])
        data = np.array(data_dict['rating_matrix'])
        np.random.seed(seed)
        indices_users = np.random.choice(range(data.shape[0]), 500, replace=False)
        indices_items = np.random.choice(range(data.shape[1]),300,replace = False)
        # 调用upcc方法
        upcc_algorithm = upcc(raw_data,data,top)
        upcc_algorithm.caculate_similarity_matrix()
        upcc_mae, upcc_rmse, upcc_failure, upcc_coverage, upcc_predicted = upcc_algorithm.evaluate(indices_users)
        # print(upcc_mae)
        # 调用ipcc方法
        ipcc_algorithm = ipcc(raw_data,data,top)
        ipcc_algorithm.calculate_pearson_similarity_matrix()
        ipcc_mae, ipcc_rmse, ipcc_failure, ipcc_coverage, ipcc_predicted = ipcc_algorithm.evaluate(indices_items)
        # print(ipcc_mae)
        # 调用trust方法
        trust_algorithm = trust(raw_data,data,top)
        trust_mae, trust_rmse, trust_failure, trust_coverage, trust_predicted = trust_algorithm.evaluate(indices_users)
        # print(trust_mae,trust_rmse,trust_failure,trust_coverage)

        # 加载三个方法的权重
        user_weight = np.zeros(data.shape)
        item_weight = np.zeros(data.shape)
        trust_weight = np.zeros(data.shape)
        user_weight[:, :] = k1
        item_weight[:, :] = k2
        trust_weight[:, :] = k3

        return upcc_predicted, ipcc_predicted, trust_predicted,user_weight,\
               item_weight, trust_weight, data, raw_data, indices_users



class combination_for_different_weight:
    def __init__(self, upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
                 item_weight, trust_weight, data, raw_data, indices_users):
        self.upcc_predicted = upcc_predicted
        self.ipcc_predicted = ipcc_predicted
        self.trust_predicted = trust_predicted
        self.user_weight = user_weight
        self.item_weight = item_weight
        self.trust_weight = trust_weight
        self.data = data
        self.raw_data = raw_data
        self.indices = indices_users

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


if __name__ == '__main__':
    upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
    item_weight, trust_weight, data, raw_data, indices_users =   generate_predicted_of_three_method(1)
    algorithm = combination_for_different_weight(  upcc_predicted, ipcc_predicted, trust_predicted, user_weight, \
    item_weight, trust_weight, data, raw_data, indices_users )
    mae, rmse, failureRate, coverage = algorithm.evaluate()
    print('mae= ', mae, 'rmse= ', rmse, 'failureRate= ', failureRate, 'coverage= ', coverage)






