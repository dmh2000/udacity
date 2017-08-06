from sklearn.naive_bayes import GaussianNB
import numpy as np

pc = 0.5
ps = 0.5

pdealc = 0.8
plifec = 0.1
plovec = 0.1

pdeals = 0.2
plifes = 0.3
ploves = 0.5

# =======
# JOINT
# =======

# love deal
pldc = plovec * pdealc * pc
plds = ploves * pdeals * ps

# life deal
# pldc = plifec * pdealc * pc
# plds = plifes * pdeals * ps

print(pldc, plds)

# ===========
# NORMALIZE
# ===========
n = pldc + plds
print(n)

# ===========
# POSTERIOR
# ===========
pcld = pldc / n
psld = plds / n

print(pcld, psld)
