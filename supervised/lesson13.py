# P(h)
h = 0.008
# P(pos | d)
Pposd = 0.98
# P(neg | d)
Pnegd = 0.02
# P(neg | nd)
Pnegnd = 0.97
# P(pos | nd)
Pposnd = 0.03

# P(H | D) = P(D | H) * P(H)  (probability of D and H together)
#           ---------------
#              P(D)           (normalizer)

# P(d | pos) = P(pos | d) * P(d)
#                   P(pos)

tp = h * Pposd  # P(+ | h) * P(h)
fn = h * Pnegd  # P(- | h) * P(h)
tn = (1.0 - h) * Pnegnd  # P(- | nh) * P(1-h)
fp = (1.0 - h) * Pposnd  # P(+ | nh) * P(1-h)

print(tp)
print(fn)
print(tn)
print(fp)

# probability of true positive, normalized
# P(h | +) = P(+ | h) * P(h) / n
TP = tp / (tp + fp)
print(tp, TP)

# probability of false positive, normalized
# P(nh | +) = P(+ | nh) * P(nh)
FP = fp / (tp + fp)
print(fp, FP)

# probability of true negative, normalized
# P(nh | -) = P(- | nh) * P(nh)
TN = tn / (tn + fn)
print(tn, TN)

# probability of false negative, normalized
# P(h | -) = P(- | h) * P(h)
FN = fn / (tn + fn)
print(fn, FN)

from sklearn.metrics import mean_squared_error

ax = [1, 0, 5, 2, 1, 4]
p1 = [1, 3, 6, 1, 2, 4]              # x % 9
print(mean_squared_error(ax, p1))
p2 = [0.34, 1, 2, 3.34, 3.67, 4.34]  # x/3
print(mean_squared_error(ax, p2))
p3 = [2, 2, 2, 2, 2, 2]              # 2
print(mean_squared_error(ax, p3))


