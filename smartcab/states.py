import re
import sys
import numpy as np

valid_actions = ['forward', 'right', 'None', 'left']

xerror = 0
invalid_cmd = 0
invalid_act = 0
mismatch = 0

# ('left', 'green', 'forward', 'left')
#  -- forward : 0.49
#  -- right : 0.00
#  -- None : -2.37
#  -- left : 1.01

def parse_state(lines):
    # line[0] = ('left', 'green', 'forward', 'left')
    m1 = re.search("\((\w+), (\w+), (\w+|None), (\w+|None)", str.replace(lines[0], "'", ""))
    # line[1] = -- forward : 0.49
    m2 = re.search("-- forward : ([0-9\.-]+)", lines[1])
    # line[2] = -- right : 0.00
    m3 = re.search("-- right : ([0-9\.-]+)", lines[2])
    # line[3] = -- None : -2.37
    m4 = re.search("-- None : ([0-9\.-]+)", lines[3])
    # line[4] = -- left : 1.01
    m5 = re.search("-- left : ([0-9\.-]+)", lines[4])
    state = dict()

    state['state'] = (m1.group(1), m1.group(2), m1.group(3), m1.group(4))
    #                   forward     right       none        left
    state['actions'] = [m2.group(1), m3.group(1), m4.group(1), m5.group(1)]
    return state


def valid_wpt(action, light, left, oncoming):
    action_okay = True
    if action == 'right':
        if light == 'red' and left == 'forward':
            action_okay = False
    elif action == 'forward':
        if light == 'red':
            action_okay = False
    elif action == 'left':
        if light == 'red' or (oncoming == 'forward' or oncoming == 'right'):
            action_okay = False
    return action_okay


def check_state(lnum, data):
    global xerror
    global invalid_cmd
    global invalid_act
    global mismatch
    global xok
    wpt = data['state'][0]
    lgt = data['state'][1]
    lft = data['state'][2]
    onc = data['state'][3]
    cmd_valid = valid_wpt(wpt, lgt, lft, onc)
    amax = np.argmax(data['actions'])
    act  = valid_actions[amax]
    act_valid = valid_wpt(act,lgt,lft,onc)
    cmd_act = wpt == act
    if not (cmd_valid or act_valid):
        prefix = "**"
        xerror += 1
    elif not cmd_valid:
        prefix = "*c"
        invalid_cmd += 1
    elif not act_valid:
        prefix = "*a"
        invalid_act += 1
    elif not cmd_act:
        prefix = "!="
        mismatch += 1
    else:
        prefix = "ok"
        xok += 1

    print prefix,
    print ',state:', data['state'],
    print ',command:', wpt,
    print ',valid:', cmd_valid,
    print ',best_action:', act,
    print ',valid:', act_valid,
    print ',match:', cmd_act,
    print ',forward:', data['actions'][0],
    print ',right:', data['actions'][1],
    print ',None:', data['actions'][2],
    print ',left:', data['actions'][3]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        fname = "logs/sim_improved-learning.txt"
    else:
        fname = sys.argv[1]
    f = open(fname, "r")

    states = 0
    xerror = 0
    invalid_cmd = 0
    invalid_act = 0
    mismatch = 0
    xok = 0
    lines = []
    i = 0
    # skip first 4 lines
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()

    # data starts here
    line = f.readline()
    lnum = 1
    while line != "":
        lines.append(line)
        i += 1
        if i == 6:
            p = parse_state(lines)
            check_state(lnum, p)
            i = 0
            lines = []
            states += 1
        line = f.readline()
        lnum += 1
    print 'states:', states,
    print ',ok:', xok,
    print ',invalid cmd:', invalid_cmd,
    print ',invalid_action:', invalid_act,
    print ',act-cmd mismatch:', mismatch
