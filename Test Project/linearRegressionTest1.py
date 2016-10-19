import numpy as np
import csv
import matplotlib.pyplot as plt

##Mean absolute Error 659.46
with open('../normalizedDataWithSharesReducedAttributes.csv', 'r') as f:
  reader = csv.reader(f)
  csv_x = list(reader)


#csv_x = np.delete(csv_x, (0), axis=0)
csv_x =  np.asarray(csv_x)
csv_y = csv_x[:, -1]

# print csv_y
#csv_x = np.delete(csv_x, (0), axis=1)
csv_x = np.delete(csv_x, (-1), axis=1)

csv_x = np.array(csv_x).astype(np.float)
csv_y = np.array(csv_y).astype(np.float).reshape(-1, 1)

with open('../thetav5.csv', 'r') as t:
    reader = csv.reader(t)
    theta = list(reader)

theta = np.array(theta).astype(np.float).reshape(-1, 1)
dot = csv_x.dot(theta)
print dot
error = abs(csv_y - dot)
plt.scatter(csv_y, dot)

totalErr = 0
for i in range(15000, 20000):
    totalErr += error[i][0]
    #totalErr += error[i][0]/csv_y[i][0]

print totalErr/5000
plt.show()
