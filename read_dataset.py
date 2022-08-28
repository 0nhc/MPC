import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow 


def calc_yaw(x, y):
    if(len(x)!=len(y)):
        print('error in function: calc_yaw')
        return
    else:
        yaw = []
        for i in range(len(x)-1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            yaw.append(math.atan2(dy, dx))
        yaw.append(yaw[-1])
        return yaw

#database = open(r'dict.eval_exp_path0.pickle','rb')
database = open(r'video3_dict.eval_exp_path0.pickle','rb')
data = pickle.load(database)
print(data['yizhuang#1/1650251580.01-1650251760.00']['rst'])
'''
path1 = np.array(data[b'364c19730f220279']['rst'][0])
path1_x = path1[:,0]
path1_y = path1[:,1]
path1_yaw = calc_yaw(path1_x,path1_y)

plt.scatter(path1_x, path1_y,c='red')
plt.show()
'''
