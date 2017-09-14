import numpy as np

def updateQ(s, ns, gamma, r, q, cols):
    # get all actions for the next state
    actions = []
    # get index of actions for each valid action in the next-state row
    [actions.append(i) for i in range(cols) if r[ns][i] >= 0]

    # get max q from the actions
    maxq = 0
    for a in actions:
        maxq = np.maximum(maxq, q[ns, a])

    q[s, ns] = r[s, ns] + gamma * maxq

    return ns


def normalizeQ(q, rows, cols):
    m = np.max(q)
    for i in range(rows):
        for j in range(cols):
            q[i, j] = q[i, j] / m


rx = [
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
]

r = np.array(rx)

print(r)

q = np.zeros(r.shape)

print(q)

gamma = 0.8
rows = 6
cols = 6
goal = 5

# training episodes
for i in range(1000):
    # select random initial state
    s = np.random.randint(0, rows)

    # while goal not reached
    while True:
        # get a random valid next state
        ns = -1
        while ns == -1:
            # get random index
            a = np.random.randint(0, 6)
            # if action at that index is valid
            if r[s][a] != -1:
                # return the index
                ns = a

        # update the q matrix
        updateQ(s, ns, gamma, r, q, cols)
        s = ns
        if s == goal:
            break
print(q)

normalizeQ(q, rows, cols)

print(q)

for s in range(rows):
    print(s)
    while s != goal:
        s = np.argmax(q[s])
        print(s)
    print()
