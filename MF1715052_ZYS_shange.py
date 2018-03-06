import numpy as np
import random
import matplotlib.pyplot as plt

alpha=0.3
gamma=0.95

# 共25种状态（x,y），见状态-坐标的txt，上下左右0,1,2,3四个动作
'''
(0,0)(0,1)(0,2)(0,3)(0,4)
(1,4)(1,3)(1,2)(1,1)(1,0)
(2,0)(2,1)(2,2)(2,3)(2,4)
(3,4)(3,3)(3,2)(3,1)(3,0)
(4,0)(4,1)(4,2)(4,3)(4,4)
'''

#不能走奖励-100，走到（4，4）奖励10，其他奖励-1
#状态为行，动作为列，上下左右0，1，2，3
reward_mat=np.array([[-1,-100,-100,-100],[-1,-1,-100,-100],[-1,-1,-100,-1],[-1,-1,-100,-1],[-100,-1,-100,-1],[-100,-1,-1,-100],[-1,-1,-1,-100],[-1,-1,-1,-1],[-1,-1,-100,-1],[-1,-100,-100,-1],[-1,-100,-1,-100],[-1,-1,-1,-100],[-1,-1,-1,-1],[-1,-1,-100,-1],[-100,-1,-100,-1],[-100,-1,-1,10],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-100,-1],[-1,-100,-100,-1],[-1,-100,-1,-100],[-1,-1,-1,-100],[-1,-1,-1,-100],[10,-1,-1,-100],[-100,-1,-1,-100]])
'''reward=np.array([[-1,-100,-100,-100],
                 [-1,-1,-100,-100],
                 [-1,-1,-100,-1],
                 [-1,-1,-100,-1],
                 [-100,-1,-100,-1],#第一列
                 [-100,-1,-1,-100],
                 [-1,-1,-1,-100],
                 [-1,-1,-1,-1],
                 [-1,-1,-100,-1],
                 [-1,-100,-100,-1],#第二列
                 [-1,-100,-1,-100],
                 [-1,-1,-1,-100],
                 [-1,-1,-1,-1],
                 [-1,-1,-100,-1],
                 [-100,-1,-100,-1],#第三列
                 [-100,-1,-1,10],
                 [-1,-1,-1,-1],
                 [-1,-1,-1,-1],
                 [-1,-1,-100,-1],
                 [-1,-100,-100,-1],#第四列
                 [-1,-100,-1,-100]
                 [-1,-1,-1,-100],
                 [-1,-1,-1,-100],
                 [10,-1,-1,-100],
                 [-100,-1,-1,-100]])#第五列'''


#q矩阵,初始为0矩阵
#状态转移矩阵，行为现状态，列为上下左右，值为转移的下一步状态
#状态标记为0~24，-1表示无效转移
trans_mat=np.array([[1,-1,-1,-1],[2,0,-1,-1],[3,1,-1,7],[4,2,-1,6],[-1,3,-1,5],[-1,6,4,-1],[5,7,3,-1],[6,8,2,12],[7,9,-1,11],[8,-1,-1,10],[11,-1,9,-1],[12,10,8,-1],[13,11,7,17],[14,12,-1,16],[-1,13,-1,15],[-1,16,14,24],[15,17,13,23],[16,18,12,22],[17,19,-1,21],[18,-1,-1,20],[21,-1,19,-1],[22,20,18,-1],[23,21,17,-1],[24,22,16,-1],[-1,23,15,-1]])
'''trans_mat=np.array([[1,-1,-1,-1],
                    [2,0,-1,-1],
                    [3,1,-1,7],
                    [4,2,-1,6],
                    [-1,3,-1,5],#第一列
                    [-1,6,4,-1],
                    [5,7,3,-1],
                    [6,8,2,12],
                    [7,9,-1,11],
                    [8,-1,-1,10],#第二列
                    [11,-1,9,-1],
                    [12,10,8,-1],
                    [13,11,7,17],
                    [14,12,-1,16],
                    [-1,13,-1,15],#第三列
                    [-1,16,14,24],
                    [15,17,13,23],
                    [16,18,12,22],
                    [17,19,-1,21],
                    [18,-1,-1,20],#第四列
                    [21,-1,19,-1],
                    [22,20,18,-1],
                    [23,21,17,-1],
                    [24,22,16,-1],
                    [-1,23,15,-1]])#第五列'''


#动作矩阵，行为25个状态，列为可进行的动作上下左右0，1，2，3
#应该根据reward矩阵代码生成
action_mat=np.array([[0],[0,1],[0,1,3],[0,1,3],[1,3],[1,2],[0,1,2],[0,1,2,3],[0,1,3],[0,3],[0,2],[0,1,2],[0,1,2,3],[0,1,3],[1,3],[1,2,3],[0,1,2,3],[0,1,2,3],[0,1,3],[0,3],[0,2],[0,1,2],[0,1,2],[0,1,2],[1,2]])
'''action_mat=np.array([[0],
                     [0,1],
                     [0,1,3],
                     [0,1,3],
                     [1,3],#第一列
                     [1,2],
                     [0,1,2],
                     [0,1,2,3],
                     [0,1,3],
                     [0,3],#第二列
                     [0,2],
                     [0,1,2],
                     [0,1,2,3],
                     [0,1,3],
                     [1,3],#第三列
                     [1,2,3],
                     [0,1,2,3],
                     [0,1,2,3],
                     [0,1,3],
                     [0,3],#第四列
                     [0,2],
                     [0,1,2],
                     [0,1,2],
                     [0,1,2],
                     [1,2]])#第五列'''
#根据最优状态转移列表和状态-坐标文本文档数据画出路径散点图
def print_path(state_list):
    data=np.loadtxt('statelocation.txt')
    for i in state_list:
        plt.plot(data[i,1],data[i,2],'ro')
    plt.xlabel('x      (interations:150)')
    plt.ylabel('y')
    plt.xlim(xmin=-0.5,xmax=4.5)
    plt.ylim(ymin=-0.5,ymax=4.5)
    plt.annotate("start", xy = (0,0), xytext = (0.5,0.5), arrowprops = dict(facecolor = 'black', shrink = 0.1))
    plt.annotate("finish", xy = (4,4), xytext = (3,3.5), arrowprops = dict(facecolor = 'black', shrink = 0.1))
    plt.show()
    


#根据q_mat打印最优状态转移列表
def print_state_list(q_mat,trans_mat):
    state_list=[0]
    state=0
    while state !=24:        
        action_jump=np.argmax(q_mat[state,:])#同一状态下最大q对应动作
        state_jump=trans_mat[state][action_jump]
        state=state_jump
        state_list.append(state)
    print('state_list',state_list)
    print_path(state_list)


#找到最后的q矩阵
def find_q_mat(action_mat,trans_mat,reward_mat):
    q_last=np.zeros([25,4])
    q_mat=np.zeros([25,4])
    for i in range(150):#迭代次数
        start_state=0
        current_state=start_state
        q_last=q_mat.copy()#保存前一次的q
        #        print('last',q_last)
        while current_state != 24:
            action=random.choice(action_mat[current_state])
            next_state=trans_mat[current_state][action]
            future_rewards=[]
            for action_next in action_mat[next_state]:
                future_rewards.append(q_mat[next_state][action_next])#找到下一个状态各个动作对应的q值
            q_state=(1-alpha)*q_mat[current_state][action]+ alpha*(reward_mat[current_state][action]+gamma*max(future_rewards))
            q_mat[current_state][action]=q_state
            current_state=next_state

    #判断q矩阵是否稳定
    for m in range(25):
        for n in range(4):
            if(abs(q_last[m,n]-q_mat[m,n])>0.0001):
                print('不合格')
                find_q_mat(action_mat,trans_mat,reward_mat)

    #打印结果			
    print('final q_mat:',q_mat)
    print_state_list(q_mat,trans_mat)

    
#    print('last q_mat:',q_last)



find_q_mat(action_mat,trans_mat,reward_mat)



















