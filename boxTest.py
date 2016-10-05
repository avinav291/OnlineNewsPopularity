import numpy as np
import csv
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

with open('../OnlineNewsPopularity boxplot.csv', 'r') as f:
  reader = csv.reader(f)
  csv_x = list(reader)


csv_x = np.delete(csv_x, (0), axis=0)
csv_x =  np.asarray(csv_x)
csv_y = csv_x[:, 20]

print csv_y
csv_x = np.delete(csv_x, (0), axis=1)
csv_x = np.delete(csv_x, (-1), axis=1)

csv_x = np.array(csv_x).astype(np.float)
csv_y = np.array(csv_y).astype(np.float).reshape(-1, 1)


r = plt.boxplot(csv_y)
topPoints = r["fliers"][0].get_data()[1]
print topPoints.size

whiskers = r["whiskers"]
print whiskers[0].get_data()
print whiskers[1].get_data()
# print np.amin(topPoints)
plt.show()