from tabnanny import verbose
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
model.add(Dense(35))
model.add(Dense(70))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(35))
model.add(Dense(65))
model.add(Dense(55))
model.add(Dense(45))
model.add(Dense(15))
model.add(Dense(1))


import time
#3. 컴파일 ,훈련
model.compile(loss='mae',optimizer='adam')
start_time = time.time()                         
print(start_time)                       # 1656033216.6214793
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)
end_time = time.time() - start_time

print("걸린시간: ", end_time)

"""
verbose 0 걸린시간 :  13.367165803909302 / 출력없다.
verbose 1 걸린시간 :  15.624984741210938 / 잔소리많다
verbose 2 걸린시간 :  13.655978679656982 / 프로그래스바 없다.
verbose 3,4,5... 걸린시간 :  13.213130474090576 / epoch만 나온다.


"""



# loss : 25.831998825073242
# r2스코어: 0.6972538615780728