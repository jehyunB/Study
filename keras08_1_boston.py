from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7, shuffle=True, random_state=20)

# print(x)
# print(y)
# print(x.shape, y.shape) # (506, 13) (506,)

# print(datasets.feature_names)  #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(datasets.DESCR)

# [실습] 아래를 완성할것
#1. train 0.7
#2. R2 0.8 이상

#2. 모델구성

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(70))
model.add(Dense(85))
model.add(Dense(65))
model.add(Dense(45))
model.add(Dense(35))
model.add(Dense(25))
model.add(Dense(1))



#3. 컴파일 ,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

#4. 평가, 에측
loss= model.evaluate(x_test, y_test)
print('loss :',loss)
y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2 =r2_score(y_test, y_predict)
print('r2스코어:', r2)

# loss : 25.831998825073242
# r2스코어: 0.6972538615780728