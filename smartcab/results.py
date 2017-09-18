import pandas as pd
import ast
import sys

def read_data(fname):
    data = pd.read_csv(fname)

    if len(data) < 10:
        print "Not enough data collected to create a visualization."
        print "At least 20 trials are required."
        return

    # Create additional features
    data['average_reward'] = (data['net_reward'] / (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['reliability_rate'] = (data['success'] * 100).rolling(window=10, center=False).mean()  # compute avg. net reward with window=10
    data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    data['good'] = (data['good_actions'] * 1.0 / \
                    (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['minor'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[1]) * 1.0 / \
                     (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['major'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[2]) * 1.0 / \
                     (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['minor_acc'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[3]) * 1.0 / \
                         (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['major_acc'] = (data['actions'].apply(lambda x: ast.literal_eval(x)[4]) * 1.0 / \
                         (data['initial_deadline'] - data['final_deadline'])).rolling(window=10, center=False).mean()
    data['epsilon'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['e'])
    data['alpha'] = data['parameters'].apply(lambda x: ast.literal_eval(x)['a'])

    # Create training and testing subsets
    training_data = data[data['testing'] == False]
    testing_data  = data[data['testing'] == True]

    return testing_data, len(training_data)

def calculate_safety(data):
    """ Calculates the safety rating of the smartcab during testing. """

    good_ratio = data['good_actions'].sum() * 1.0 / \
                 (data['initial_deadline'] - data['final_deadline']).sum()

    if good_ratio == 1:  # Perfect driving
        return ("A+", "green")
    else:  # Imperfect driving
        if data['actions'].apply(lambda x: ast.literal_eval(x)[4]).sum() > 0:  # Major accident
            return ("F", "red")
        elif data['actions'].apply(lambda x: ast.literal_eval(x)[3]).sum() > 0:  # Minor accident
            return ("D", "#EEC700")
        elif data['actions'].apply(lambda x: ast.literal_eval(x)[2]).sum() > 0:  # Major violation
            return ("C", "#EEC700")
        else:  # Minor violation
            minor = data['actions'].apply(lambda x: ast.literal_eval(x)[1]).sum()
            if minor >= len(data) / 2:  # Minor violation in at least half of the trials
                return ("B", "green")
            else:
                return ("A", "green")


def calculate_reliability(data):
    """ Calculates the reliability rating of the smartcab during testing. """

    success_ratio = data['success'].sum() * 1.0 / len(data)

    if success_ratio == 1:  # Always meets deadline
        return ("A+", "green")
    else:
        if success_ratio >= 0.90:
            return ("A", "green")
        elif success_ratio >= 0.80:
            return ("B", "green")
        elif success_ratio >= 0.70:
            return ("C", "#EEC700")
        elif success_ratio >= 0.60:
            return ("D", "#EEC700")
        else:
            return ("F", "red")


def count_states(filename):
    f = open(filename,'r')
    states = 0
    for line in iter(f):
        # check for '(' which is only once per state
        if line[0] == '(':
            states += 1
    return states


if len(sys.argv) < 2:
    datafile = 'sim_improved-learning.csv'
else:
    datafile = sys.argv[1]

if len(sys.argv) < 3:
    statefile = 'sim_improved-learning.csv'
else:
    statefile = sys.argv[2]

test_data,trials = read_data(datafile)
reliability = calculate_reliability(test_data)
safety = calculate_safety(test_data)
states = count_states(statefile)

print "reliability,",
print reliability[0],
print ',safety,',
print safety[0],
print ',trials,',
print trials,
print ',states,',
print states
