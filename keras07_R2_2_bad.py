#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지마
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. bach_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상
# [실습 시작]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
 
 #1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,8,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.7, shuffle=True, random_state=20)

#2. 모델구성
model = Sequential()
model.add(Dense(99, input_dim=1))
model.add(Dense(8))
model.add(Dense(59))
model.add(Dense(91))
model.add(Dense(28))
model.add(Dense(2))
model.add(Dense(50))
model.add(Dense(5))
model.add(Dense(75))
model.add(Dense(15))
model.add(Dense(90))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 에측
loss= model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict= model.predict(x)

from sklearn.metrics import r2_score
r2 =r2_score(y, y_predict)
print('r2스코어:', r2)

# loss : 3.592268228530884
# r2스코어: 0.36222615796645075
# loss 'mse' 에서 'mae' 로 바꿨다
# 레이어의 개수를 늘렸다









# import matplotlib.pyplot as pit

#  pit.scatter(x,y)
# pit.plot(x,y_predict, color='red')
# pit.show()
