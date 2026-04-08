import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./points50k.csv')
#print(data.head())

x = data['x']
y = data['y']
print(len(x)) # 50000
#plt.scatter(x,y,s=0.1)
#plt.show()

import torch
'''Tensor 메모리 + 메타데이터 포함
 ├── Storage (실제 데이터 메모리)
 ├── Shape (size)
 ├── Stride (메모리 접근 방식)
 ├── dtype (float32 등)
 ├── device (CPU / GPU)
'''

xtensor = torch.tensor(x, dtype=torch.float32)
ytensor = torch.tensor(y, dtype=torch.float32)

data_len = len(xtensor) # 50000
splitn = int(data_len*0.9)
trainx = xtensor[:splitn]
trainy = ytensor[:splitn]
valx = xtensor[splitn:]
valy = ytensor[splitn:]

def get_batch(x,y,batchsize=16):
    datalen = len(x)
    indices = torch.randint(0, datalen, (batchsize,))
    #print(indices)
    return x[indices], y[indices]
    #16개 index값의 각각의 x,y값을 말함 16개 다 들어있다.

#batchx, batchy = get_batch(trainx, trainy)
#print(batchx) # 16개만한번 test로 해보는거지.

#model은 1차 직선방정식. ax+b = 0
a = torch.tensor(3.0, requires_grad = True)
b = torch.tensor(0.0, requires_grad = True)

#이 직선방정식의 lossfn
for step in range(5000):
    batchx, batchy = get_batch(trainx, trainy)
    diff = (batchy - (a*batchx + b))**2
    loss = torch.mean(diff) # 16개 평균

    valloss = torch.mean(valy -(a*valx +b))**2 #valdata 전부
    if step%100 ==0: print(f"{step}, tloss:{loss} vloss:{valloss}, a:{a.item()}, b:{b.item()})")

    grad = loss.backward()

    with torch.no_grad():# weight update은 학습 안해도됨.
        a -= 0.001* a.grad
        b -= 0.001* b.grad
    a.grad.zero_() # 위에 grad 쓰고 다시 0으로 초기화
    b.grad.zero_() # 동상
    
import numpy as np

plt.scatter(x,y,s=0.1)
aa = a.item()
bb = b.item()

x_line = np.linspace(-10, 10, 100)
y_line = aa * x_line + bb

plt.plot(x_line, y_line, color='y', lw=3)
plt.show()