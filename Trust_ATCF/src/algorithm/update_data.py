import numpy as np
import json

class update:
    def __init__(self):
        self.raw_data = np.array([[5, 3, 4, 0], [5, 4, 3, 1], [5, 3, 4, 0]])
        self.data = np.copy(self.raw_data)
        self.data[0][1] = 0
        self.data[1][0] = 0
        self.data[0][2] = 0
        self.data[1][1] = 0
        self.data[1][2] = 0
        self.trust_matrix = np.array([[0, 0, 1], [0, 0, 0], [1, 1, 0]])
        (self.user_count, self.item_count) = self.data.shape
        print(self.data)
        print(self.trust_matrix)

    def iteration(self, times, a):
        main_matrix = a*self.data
        trust_nums = np.sum(self.trust_matrix,axis=1).reshape(self.user_count,1)
        trust_nums[trust_nums==0] = 1
        addition_matrix = np.dot( self.trust_matrix, self.data)
        # print(matrix)
        addition_matrix = (1-a)*addition_matrix/trust_nums
        print(main_matrix)
        print(addition_matrix)
        matrix = main_matrix + addition_matrix
        print(matrix)




if __name__ == '__main__':
    newdata = update()
    newdata.iteration(1,0.8)