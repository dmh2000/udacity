import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import sys
from time import time


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, epsilon_decay=1, epsilon_type='default'):
        super(LearningAgent, self).__init__(env)  # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning  # Whether the agent is expected to learn
        self.Q = dict()  # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon  # Random exploration factor
        self.alpha = alpha  # Learning factor

        # Set any additional class parameters as needed
        self.testing = False
        self.epsilon_decay = epsilon_decay
        self.epsilon_type = epsilon_type
        self.t = 0

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        # increment trial count
        self.t += 1

        # update testing flag
        self.testing = testing

        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            # testing
            self.epsilon = 0.0
            self.alpha = 0.0
        elif self.learning:
            # learning
            # linear
            if self.epsilon_type == 'default':
                # linear
                self.epsilon -= self.epsilon_decay
            elif self.epsilon_type == 'improved-linear':
                self.epsilon -= self.epsilon_decay
            elif self.epsilon_type == 'a^t':
                # concave curve : below linear
                self.epsilon = self.epsilon_decay ** self.t
            else:
                # default is linear
                self.epsilon -= self.epsilon_decay
        else:
            # not learning
            self.epsilon = 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint()  # The next waypoint
        inputs = self.env.sense(self)  # Visual input - intersection light and traffic
        # deadline = self.env.get_deadline(self)  # Remaining deadline

        # NOTE : you are not allowed to engineer eatures outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.

        # Set 'state' as a tuple of relevant data for the agent        
        # state = (waypoint, inputs['light'], inputs['left'], inputs['oncoming'])
        state = (waypoint, inputs['light'], inputs['left'], inputs['oncoming'])

        return state

    # check that an action is legal for the current sense inputs
    @staticmethod
    def valid_action(action, inputs):
        light = inputs['light']
        left = inputs['left']
        oncoming = inputs['oncoming']

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

    # get a random action that is legal for the current sense inputs
    def random_action(self, actions, inputs):
        # only pick from legal actions
        legal_actions = []

        # accumulate all legal actions given the inputs
        for action in actions:
            action_okay = self.valid_action(action, inputs)
            if action_okay:
                # add this one to list
                legal_actions.append(action)

        # there will always be at least 'None' as a valid action
        action = np.random.choice(legal_actions)

        return action

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        # Calculate the maximum Q-value of all actions for a given state

        # get the maximum value from the current state
        # in this case, maximum of each possible action
        if state in self.Q:
            maxQ = np.max(self.Q[state].values())
        else:
            maxQ = 0

        return maxQ

    def get_maxQaction(self, state):
        # get maxq value
        maxq = self.get_maxQ(state)

        # get list of actions equal to maxq value
        # this is the 'summation' stage although in this case it only looks back one
        a = []
        for key, value in self.Q[state].iteritems():
            if value == maxq:
                a.append(key)

        # in case floating point comparison error
        assert len(a) > 0

        # select a random action from list
        action = np.random.choice(a)

        return action

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning:
            if state not in self.Q:
                q = dict()
                for action in self.valid_actions:
                    q[action] = 0.0
                self.Q[state] = q

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)  # Visual input - intersection light and traffic
        action = None

        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        # THIS VERSION ONLY ALLOWS CHOICE OF ACTIONS THAT FOLLOW THE TRAFFIC RULES, TO AVOID ACCIDENTS
        if self.testing:
            # testing, get action based on learned maxq
            action = self.get_maxQaction(state)
        elif self.learning:
            # choose an action
            p = np.random.random()
            # if p is less than epsilon,
            # then explore
            # else choose best
            # as epsilon gets smaller, chance of exploring decreases
            if p < self.epsilon:
                # choose a random action that is valid
                action = self.random_action(self.valid_actions, inputs)
            else:
                # choose action with highest q value
                action = self.get_maxQaction(state)
        else:
            # not testing or learning
            # choose a random action that is valid
            action = self.random_action(self.valid_actions, inputs)

        return action

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        # THIS WORKS (with enough learning trials) BUT DOESN'T SEEM TO MATCH Q-LEARNING
        # IT DOES NOT USE THE NEXT STATE-ACTIONS FOR ITS UPDATE
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            # apply an alpha filter to the current state and its reward
            # larger alpha = more reward, smaller alpha = more current Q value
            self.Q[state][action] = (1.0 - self.alpha) * self.Q[state][action] + self.alpha * reward

        return

    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()  # Get current state
        self.createQ(state)  # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action)  # Receive a reward
        self.learn(state, action, reward)  # Q-learn

        return


def run(test_args):
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,
                             learning=test_args['learning'],
                             epsilon=test_args['epsilon'],
                             alpha=test_args['alpha'],
                             epsilon_decay=test_args['epsilon_decay'],
                             epsilon_type=test_args['epsilon_type'])

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, optimized=test_args['optimized'], update_delay=0.001, log_metrics=True, display=False)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=test_args['n_test'], tolerance=test_args['tolerance'])


if __name__ == '__main__':
    # arguments for improvements
    test_args = [
        # sim, no learning parameters
        {'learning': False, 'optimized': False, 'alpha': 0.5, 'n_test': 10,
         'epsilon': 1.0, 'tolerance': 0.05, 'epsilon_decay': 0.05, 'epsilon_type': 'no-learning'},

        # default-learning parameters
        {'learning': True, 'optimized': False, 'alpha': 0.5, 'n_test': 10,
         'epsilon': 1.0, 'tolerance': 0.05, 'epsilon_decay': 0.05, 'epsilon_type': 'default-linear'},

        # optimized-learning, same as default except linear decay is smaller, resulting in more trials (~9500)
        {'learning': True, 'optimized': True, 'alpha': 0.5, 'n_test': 20,
         'epsilon': 1.0, 'tolerance': 0.05, 'epsilon_decay': 0.0005, 'epsilon_type': 'improved-linear'},

        # optimized-learning, epsilon = a^t
        {'learning': True, 'optimized': True, 'alpha': 0.5, 'n_test': 20,
         'epsilon': 1.0, 'tolerance': 0.05, 'epsilon_decay': 0.996, 'epsilon_type': 'a^t'}
    ]

    if len(sys.argv) > 1:
        v = int(sys.argv[1])
    else:
        # perform optimized-learning, epsilon = a^t
        v = 3

    t0 = time()
    run(test_args[v])
    t1 = time()
    # elapsed time
    print 'runtime',
    print t1 - t0


# count number of states generated from  <log>.txt file
# grep \( logs\sim_default-learning.txt  | wc
