#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y, epochs=2000)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([4])
print('6의 예측값 :', result)

# loss : 0.41046515107154846
# 6의 예측값 : [[5.9964366]] 
