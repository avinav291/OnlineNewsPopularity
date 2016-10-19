import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn


with open('../8 Oct 2016/OnlineNewsPopularity boxplot.csv', 'r') as f:
  reader = csv.reader(f)
  csv_x = list(reader)


csv_x = np.delete(csv_x, (0), axis=0)
csv_x =  np.asarray(csv_x)
csv_y = csv_x[:, -1]

# print csv_y
csv_x = np.delete(csv_x, (0), axis=1)
csv_x = np.delete(csv_x, (-1), axis=1)

csv_x = np.array(csv_x).astype(np.float)
csv_y = np.array(csv_y).astype(np.float).reshape(-1, 1)

with open('../theta v4 21k.csv', 'r') as t:
    reader = csv.reader(t)
    theta = list(reader)

theta = np.array(theta).astype(np.float).reshape(-1, 1)

dot = csv_x.dot(theta)
print dot
error = csv_y - dot
plt.scatter(csv_y, dot)
plt.show()

totalErr = 0
for i in range(15000, 20000):
    totalErr += error[i][0]/csv_y[i][0]

print totalErr/50