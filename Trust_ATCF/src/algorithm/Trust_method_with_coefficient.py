""""
该方法为upcc方法预测用户评分，且将无法预测的分数使用平均分填充
"""
import numpy as np
import json
import time

class trust:
    def __init__(self, raw_data, data, top = 5):
        self.raw_data = raw_data
        self.data = data
        (self.user_count, self.item_count) = data.shape
        load_data = json.load(open('../../dataset/preprocessed_data_for_0.2ratio.json'))
        self.trust_matrix = np.array(load_data['trust_matrix'])
        self.top = top

    def calculate_sigmoid_coefficient(self):
        """
        该方法用于求用于优化相似度矩阵的系数矩阵
        :return:
        """
        data_flag= self.data>0

        # 每两个用户共同打分的项目个数矩阵
        both_rating_num = np.zeros((self.user_count, self.user_count))
        for i in range(self.user_count):
            i_user_flag = data_flag[i]
            for j in range(i+1,self.user_count):
                j_user_flag = data_flag[j]
                both_num = np.sum(np.multiply(i_user_flag,j_user_flag))
                both_rating_num[i][j] = both_num
                both_rating_num[j][i] = both_num
        coefficient = 1/(1+np.exp(-both_rating_num))
        self.trust_matrix = self.trust_matrix*coefficient



    def findTrustUsers(self, index, top=5):
        trust_value = self.trust_matrix[index]
        return np.argsort(trust_value)[-top:], np.sort(trust_value)[-top:].reshape(1, top)

    def evaluate(self, indices):
        # 初始化
        mae = 0
        rmse = 0
        num_of_predicted = 0
        num_of_failure = 0
        # 使用predicted_matrix记录经过trust方法预测后的矩阵，且无法预测的值暂时不用矩阵填充
        predicted_matrix = np.copy(self.data)
        # 初始化所有商品均未被覆盖
        coverage_flag = [False for x in range(self.data.shape[1])]
        # 求所有用户的平均分数
        valid_flag = self.data > 0
        valid_flag_count = np.sum(valid_flag, axis=1)
        valid_flag_count[valid_flag_count == 0] = 1
        users_avg_value = np.sum(self.data, axis=1) / valid_flag_count
        # 若用户没有任何打分，则将平均数置为全局平均数
        users_avg_value[users_avg_value == 0] = np.sum(self.data) / np.sum(self.data > 0)
        users_avg_value.reshape(self.user_count, 1)

        for idx in indices:
            predicted_columns = np.multiply(self.data[idx] == 0, self.raw_data[idx] > 0)
            # 记录下需要预测的商品编号
            predicted_items = []
            for (i, flag) in enumerate(predicted_columns):
                if flag == True: predicted_items.append(i)
            predicted_items = np.array(predicted_items)
            # print('需要预测的商品是',predicted_items)
            trust_users, trust_values = self.findTrustUsers(idx, self.top)
            #           提取信任用户的评分
            data_for_predicted = self.data[trust_users, :]
            data_for_predicted = data_for_predicted[:, predicted_columns > 0]
            data_for_reference = self.raw_data[idx, predicted_columns > 0]
            data_for_predicted_flags = data_for_predicted > 0
            # 用于计算多少信任用户的数据可以被预测，且valid_counts为预测目标用户对目标项目打分时所有有效信任用户的信任分数之和
            valid_counts = np.dot(trust_values, data_for_predicted_flags).squeeze()

            predicted_values = np.zeros((1, data_for_predicted.shape[1]))
            if np.sum(valid_counts) == 0:
                # 如果这个用户的所有项都不能被预测，则置为该用户所有打分的平均分
                predicted_values[:, :] = users_avg_value[idx]
                num_of_failure += data_for_predicted.shape[1]
            else:
                num_of_failure += np.sum(valid_counts == 0)
                valid_counts[valid_counts == 0] = 1
                predicted_values = np.dot(trust_values, data_for_predicted) / valid_counts
                # 将可以预测出的商品置为True
                for (i, value) in enumerate(predicted_values[0]):
                    if value > 0:
                        if coverage_flag[predicted_items[i]] == False:
                            coverage_flag[predicted_items[i]] = True
                        predicted_matrix[idx][predicted_items[i]] = value
                predicted_values[predicted_values == 0] = users_avg_value[idx]
            predicted_values = predicted_values.squeeze()
            # print('对第%d用户的预测分数:%s' % (idx, predicted_values))

            mae += np.sum(np.abs(predicted_values - data_for_reference))
            rmse += np.sum(np.square(predicted_values - data_for_reference))
            num_of_predicted += data_for_predicted.shape[1]
            # print(coverage_flag)
        return mae / num_of_predicted, np.sqrt(rmse / num_of_predicted), num_of_failure / num_of_predicted,\
               np.sum(np.array(coverage_flag))/self.data.shape[1], predicted_matrix



if __name__ == '__main__':
    with open('../../dataset/preprocessed_data_for_0.2ratio.json', 'r') as f:
        data_dict = json.load(f)
        raw_data = np.array(data_dict['raw_matrix'])
        data = np.array(data_dict['rating_matrix'])
        trust_algorithm = trust(raw_data, data, top=10)
        trust_algorithm.calculate_sigmoid_coefficient()
        for j in range(3):
            begin=time.time()
            np.random.seed(j)
            indices = np.random.choice(range(data.shape[0]), 500, replace=False)
            mae, rmse, failureRate,coverage,predicted_matrix = trust_algorithm.evaluate(indices)
            print('mae= ',mae,'rmse= ',rmse,'failureRate= ',failureRate,'coverage= ',coverage)
            print('cost=',time.time()-begin)
            # print('predicted matrix=\n',predicted_matrix)
        # with open('../../output/datafile_itself/K_choice/more_trust_predicted_matrix_top=15.json', 'w')as fp:
        #     json.dump({'predicted matrix':predicted_matrix.tolist()},fp)

        # upcc_dict = {}
        # for top in range(5,101,5):
        #     branches = np.zeros((3,100))
        #     for i in range(100):
        #         np.random.seed(i)
        #         indices = np.random.choice(range(4000), 50, replace=False)
        #         upcc_algorithm = upcc(raw_data, data, top)
        #         matrix = upcc_algorithm.caculate_similarity_matrix()
        #         mae, rmse, failureRate =upcc_algorithm.evaluate(indices)
        #         print(mae,rmse,failureRate)
        #         branches[0, i], branches[1, i], branches[2, i] = mae, rmse, failureRate
        #     upcc_dict[top] = branches.tolist()
        # with open('../../output/datafile_itself/fill_upcc_dict_0.2ratio.json', 'w') as fp:
        #     json.dump(upcc_dict, fp)
        # print(mae, rmse, failureRate)
        # 存储皮尔逊相似度矩阵
        # with open('../../dataset/trust_value_matrix_0.2ratio_withCoefficient.json', 'w') as fp:
        #     json.dump({'trust_matrix': trust_algorithm.trust_matrix.tolist()}, fp)


