import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]
             )

y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) 
print(y.shape) # (10,2)
x = x.T
print(x)
print(x.shape)

 


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(40))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x ,y, epochs=100, batch_size=1)

#4. 평가,예측

loss = model.evaluate(x,y)
print('loss :' , loss)
result =model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값 :', result)








