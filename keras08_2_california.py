from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)  #(20640, 8) (20640,)

print(datasets.feature_names) # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets. DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.9, shuffle=True, random_state=20)

#2. 모델구성

model = Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(35))
model.add(Dense(50))
model.add(Dense(45))
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
model.fit(x_train, y_train, epochs=100, batch_size=30)

#4. 평가, 에측
loss= model.evaluate(x_test, y_test)
print('loss :',loss)
y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2 =r2_score(y_test, y_predict)
print('r2스코어:', r2)

# loss : 0.6113101243972778
# r2스코어: 0.5519585308448118