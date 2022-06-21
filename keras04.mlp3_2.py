import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10))
# for i in range(10):
#     print(i)
print(x.shape) # (3, 10)
x = np.transpose(x)
print(x.shape) # (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1 , 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            [9,8,7,6,5,4,3,2,1,0]])

y = np.transpose(y)
print(y.shape) # (10, 2)

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=3))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x ,y, epochs=500, batch_size=1)

#4. 평가,예측

loss = model.evaluate(x,y)
print('loss :' , loss)
result =model.predict([[9, 30, 210]])
print('[10, 1.9, 0]의 예측값 :', result)



# 예측 : [[9, 30 ,210]]  예상 y값 [[10, 1.9, 0]]

# loss : 0.05820118263363838
# [10, 1.9, 0]의 예측값 : [[9.476838   1.8914266  0.02817045]]
