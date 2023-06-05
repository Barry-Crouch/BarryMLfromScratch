import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from decision_tree import DecisianTreeRegressor
from utils import mae


Data = pd.read_excel('./Data1.xlsx', sheet_name='Sheet3')
Data = Data.to_numpy(dtype='float64')
X = Data[:,3:]
y= Data[:,2:3]
y2 = Data[:, 1:2]
y3 = Data[:, 0:1]



X_train, X_test , y_train, y_test= train_test_split(X , y, test_size=0.3, random_state=30)
model = DecisianTreeRegressor()
model.fit(X_train, y_train)
y_pred = np.array(model.predict(X_test))
e = mae(y_test, y_pred)
print(e)

# model2 = RF()
# X_train, X_test , y_train, y_test= tts(X , y2, ts= 0.3)
# model2.fit(X_train, y_train)
# y_pred = np.array(model2.predict(X_test))
# e2 = mae(y_test, y_pred)
# print(e2)


# model3 = RM()
# X_train, X_test , y_train, y_test= tts(X , y3, ts= 0.3)
# model3.fit(X_train, y_train)
# y_pred = np.array(model3.predict(X_test))
# e3 = mae(y_test, y_pred)
# print(e3)

# fig, ax = plt.subplots(figsize=(6, 5))
# ax.scatter(y_test, y_pred, color='purple')
# ax.set_xlabel('y_True')
# ax.set_ylabel('y_Pred')
# plt.ylim(85, 97)
# plt.xlim(85, 97)
# line = mlines.Line2D([0, 1], [0, 1], color='green')
# transform = ax.transAxes
# line.set_transform(transform)
# ax.add_line(line)
# fig.tight_layout()
# plt.show()
# plt.draw()

# fig, ax = plt.subplots(figsize=(6, 5))
# ax.scatter(y_test, y_pred, color='purple')
# ax.set_xlabel('y_True')
# ax.set_ylabel('y_Pred')
# plt.ylim(70, 120)
# plt.xlim(70, 120)
# line = mlines.Line2D([0, 1], [0, 1], color='green')
# transform = ax.transAxes
# line.set_transform(transform)
# ax.add_line(line)
# fig.tight_layout()
# plt.show()
# plt.draw()

# fig, ax = plt.subplots(figsize=(6, 5))
# ax.scatter(y_test, y_pred, color='purple')
# ax.set_xlabel('y_True')
# ax.set_ylabel('y_Pred')
# plt.ylim(1, 4.5)
# plt.xlim(1, 4.5)
# line = mlines.Line2D([0, 1], [0, 1], color='green')
# transform = ax.transAxes
# line.set_transform(transform)
# ax.add_line(line)
# fig.tight_layout()
# plt.show()
# plt.draw()


