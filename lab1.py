import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from statsmodels.distributions import ECDF

# Y = 2 / (X + 2); a = -1; b = 5
# f(y) = 1 / (3 * y^2)
# F(y) = -1 / (3 * y) + 7 / 6
a = -1
b = 5
N = 100
np.random.seed(seed=int(time.time()))


def y(x):
    return 2 / (x + 2)


def get_x(n):
    return np.random.random((n,)) * (b - a) + a


def get_ecdf(y):
    return ECDF(y)


def dist_func(y):
    if y < 2 / 7:
        return 0
    if y > 2:
        return 1
    return -1 / (3 * y) + 7 / 6


def dist_density(y):
    if y < 2 / 7 or y > 2:
        return 0
    return 1 / (3 * y ** 2)


def get_intervals_number(n):
    if n <= 100:
        return int(np.sqrt(n))
    return int(4 * np.log10(n))


def get_bins_sequence(y):
    length = len(y)
    m = get_intervals_number(length)
    v = int(N / m)
    i = 0
    result_sequence = [y[0]]
    while i < length:
        i += v
        if i < length - 1:
            result_sequence.append((y[i] + y[i + 1]) / 2)
        else:
            result_sequence.append(y[length - 1])

    return result_sequence


def grouped_ECDF(values, bins):
    def _ecdf(x):
        if x < bins[0]:
            return 0
        if x > bins[-1]:
            return 1
        for i in range(len(bins) - 1):
            if bins[i] < x <= bins[i + 1]:
                return values[i]

    return _ecdf


X = get_x(N)
Y = y(X)
Y = np.sort(Y)
ecdf = get_ecdf(Y)

t = np.arange(0.2, 2.1, 0.01)
F = np.array([dist_func(val) for val in t])
f = np.array([dist_density(val) for val in t])

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].set(xlabel='Y', ylabel='F(Y)',
       title='ECDF')
axes[0, 0].grid()
axes[0, 0].step(t, ecdf(t))
axes[0, 0].plot(t, F, '--')

axes[0, 1].grid()
axes[0, 1].set_axisbelow(True)
hist1 = axes[0, 1].hist(Y, get_intervals_number(N), density=True)
axes[0, 1].plot(t, f, '--')
axes[0, 1].set(xlabel='Y', ylabel='f(Y)',
        title='equal interval method')

axes[1, 1].grid()
axes[1, 1].set_axisbelow(True)
hist_equal_interval = axes[1, 1].hist(Y, get_intervals_number(N), density=True, cumulative=True, histtype='step')
ecdf_equal_interval = grouped_ECDF(hist_equal_interval[0], hist_equal_interval[1])
axes[1, 1].step(t, [ecdf_equal_interval(val) for val in t])
axes[1, 1].plot(t, F, '--')
axes[1, 1].set(xlabel='Y', ylabel='F(Y)',
        title='ECDF of grouped data')

axes[0, 2].grid()
axes[0, 2].set_axisbelow(True)
hist2 = axes[0, 2].hist(Y, get_bins_sequence(Y), density=1)
axes[0, 2].plot(t, f, '--')
axes[0, 2].set(xlabel='Y', ylabel='f(Y)',
        title='equiprobability method')

axes[1, 2].grid()
axes[1, 2].set_axisbelow(True)
hist_equiprobable = axes[1, 2].hist(Y, get_bins_sequence(Y), density=1, cumulative=True, histtype='step')
ecdf_equiprobable = grouped_ECDF(hist_equiprobable[0], hist_equiprobable[1])
axes[1, 2].step(t, [ecdf_equiprobable(val) for val in t])
axes[1, 2].plot(t, F, '--')

x_equal_interval = []
for i in range(len(hist1[1]) - 1):
    x_equal_interval\
        .append((hist1[1][i] + hist1[1][i + 1]) / 2)

x_equiprobable = []
for i in range(len(hist2[1]) - 1):
    x_equiprobable\
        .append((hist2[1][i] + hist2[1][i + 1]) / 2)

axes[1, 0].grid()
axes[1, 0].set_axisbelow(True)
axes[1, 0].plot(x_equal_interval, hist1[0])
axes[1, 0].plot(x_equiprobable, hist2[0])
axes[1, 0].plot(t, f, '--')

data_equal_interval = []
for i in range(len(hist1[0])):
    new_row = [str(hist1[0][i]), str(hist1[1][i]) + ' - ' + str(hist1[1][i + 1])]
    data_equal_interval.append(new_row)

print(pd.DataFrame(data=data_equal_interval, columns=['Value', 'Interval']))
print()

data_equiprobable = []
for i in range(len(hist2[0])):
    new_row = [str(hist2[0][i]), str(hist2[1][i]) + ' - ' + str(hist2[1][i + 1])]
    data_equiprobable.append(new_row)

print(pd.DataFrame(data=data_equiprobable, columns=['Value', 'Interval']))

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()
