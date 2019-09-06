import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sm


def matrix_factorize(u, p, q, K, alpha, beta):
    result = []
    step = 0
    while 1:
        step += 1
        e = 0
        for i in range(len(u)):
            for j in range(len(u[i])):
                if u[i][j]:
                    eij = u[i][j] - np.dot(p[i, :], q[:, j])
                    e += pow(u[i][j]- np.dot(p[i, :], q[:, j]), 2)
                    for k in range(K):
                        e += beta/2*(pow(p[i][k], 2)+pow(q[k][j], 2))
                    for k in range(K):
                        p[i][k] = p[i][k] + alpha*(2*eij*q[k][j]-beta*p[i][k])
                        q[k][j] = q[k][j] + alpha*(2*eij*p[i][k]-beta*q[k][j])
        result.append(e)
        print('迭代轮次：', step, '   e', e)
        if eij < 0.00001:
            break
    return p, q, result

R=[
   [5,3,1,1,4],
   [4,0,0,1,4],
   [1,0,0,5,5],
   [1,3,0,5,0],
   [0,1,5,4,1],
   [1,2,3,5,4]
   ]

R = np.random.rand(100, 200)
R = np.random.randint(0, 10, (100, 200))
R = sm.imread('qinghuaxuetang.jpg')
#R = np.array(R)
n = len(R)
m = len(R[0])
print(R)
alpha = 0.0001
beta = 0.002
k = 4
p = np.random.rand(n, k)
q = np.random.rand(k, m)

p, q, result = matrix_factorize(R, p, q, k, alpha, beta)

print('p: \n', p)
print('q: \n', q)
MF = np.dot(p, q)
print('R: \n', R)
print('MF:\n', MF)

num = len(result)
x = range(num)
plt.plot(x, result, color = 'b', linewidth = 3)
plt.xlabel('generation')
plt.ylabel('loss')
plt.show()
