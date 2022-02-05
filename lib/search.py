# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

# DFS, BFS, UCS, A* search algorithms developed by Blake Whitman for CSE 3521 @ OSU, instructed by Professor Perrault

import util
from util import heappush, heappop
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
      """
      Returns the start state for the search problem
      """
      util.raiseNotDefined()

    def isGoalState(self, state):
      """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
      util.raiseNotDefined()

    def getSuccessors(self, state):
      """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
      util.raiseNotDefined()

    def getCostOfActions(self, actions):
      """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
      util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    from util import Stack
    # DFS will expand a deepest node first. Check below for other types of implementations such as BFS, UCS, and A*
    closedset = [] # used to avoid cycle issues and infinite loops
    actions = [] # create a list of actions–this is what we are instructed to return

    ''' Next, we initialize the frontier using the provided util.py file, as the instructions hint at us to do
        Stacks are LIFO structures, which is useful for this method since that is how DFS is implemented
    '''
    frontier = Stack()
    frontier.push((problem.getStartState(), actions)) # intialize frontier with first position/empty action

    while not frontier.isEmpty():
        state, actions = frontier.pop() # gives values to STATE and ACTIONS; we use these to verify whether we have
        # visited a certain "state" before and the actions it took to arrive at a certain state
        if not state in closedset: # used to avoid duplicates and cycles
            closedset.append(state)
            if problem.isGoalState(state): # We are done, solution has been found
                print("Found goal!")
                return actions
            else: # we must go deeper, or backtrack before continuing onward
                # follows implementation in the sample code within provided powerpoint
                # next_state is a set of COORDINATES, action is a DIRECTION, cost is a NUMBER
                for next_state, action, cost in problem.getSuccessors(state):
                    next_check = (next_state, actions + [action])
                    # note that we don't care about cost here since it's a DFS algorithm
                    frontier.push(next_check)
    # Test with the following command line prompt: python3 py/pacman.py -l mediumMaze -p SearchAgent -a fn=dfs
    return actions

def breadthFirstSearch(problem):
    from util import Queue
    # BFS will search wider/shallower first. Check above for an implementation of DFS and below for A* and UCS.
    # The ONLY difference in this implementation compared to DFS is that we used a queue for the frontier instead of a
    # stack
    closedset = []  # used to avoid cycle issues and infinite loops
    actions = []  # create a list of actions–this is what we are instructed to return

    ''' Next, we initialize the frontier using the provided util.py file, as the instructions hint at us to do
        Queues are FIFO structures, which is useful for this method since that is how BFS is implemented
    '''
    frontier = Queue()
    frontier.push((problem.getStartState(), actions))  # intialize frontier with first position/empty action

    while not frontier.isEmpty():
        state, actions = frontier.pop()  # gives values to STATE and ACTIONS; we use these to verify whether we have
        # visited a certain "state" before and the actions it took to arrive at a certain state
        if not state in closedset:  # used to avoid duplicates and cycles
            closedset.append(state)
            if problem.isGoalState(state):  # We are done, solution has been found
                print("Found goal!")
                return actions
            else:  # we must go wider, or backtrack before continuing onward
                # follows implementation in the sample code within provided powerpoint
                # next_state is a set of COORDINATES, action is a DIRECTION, cost is a NUMBER
                for next_state, action, cost in problem.getSuccessors(state):
                    next_check = (next_state, actions + [action])
                    # note that we don't care about cost here since it's a BFS algorithm
                    frontier.push(next_check)
    # Test with the following command line prompt: python3 py/pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
    return actions

def uniformCostSearch(problem):
    from util import PriorityQueue
    # UCS expands the cheapest node first. Check above for DFS and BFS implementations and below for A*
    # This method is essentially A* without the heuristic portion included
    closedset = []  # used to avoid cycle issues and infinite loops
    actions = []  # create a list of actions–this is what we are instructed to return
    legal_moves = problem.getStartState() # first positioning
    ''' Next, we initialize the frontier using the provided util.py file, as the instructions hint at us to do
            UCS is implemented using a Priority Queue type structure or a Heap. In our case, we will be using a 
            PriorityQueue, which orders the nodes from cheapest to most expensive and expands them in that order
            '''
    expanded = PriorityQueue()  # This PriorityQueue is for already expanded states, while the next PriorityQueue is for
    # specifically for the fringe/frontier
    frontier = PriorityQueue()
    frontier.push(legal_moves, 0) # Push first positioning plus a 0, which in this case represents the initial cost of
    # actions
    state = frontier.pop()  # represents current state of pacman

    while not problem.isGoalState(state): # We are done, solution has been found if this is TRUE
        if not state in closedset: # cycle checking
            closedset.append(state)
            # As before, next_state is a set of COORDINATES, action is a DIRECTION, cost is a NUMBER
            for next_state, action, cost in problem.getSuccessors(state):
                successor_cost = problem.getCostOfActions(actions + [action])  # How much the successor path costs
                if next_state not in closedset:  # more cycle checking
                    frontier.push(next_state, successor_cost) # same idea as BFS/DFS
                    expanded.push(actions + [action], successor_cost)

        actions = expanded.pop()  # Updates the determined cheapest path.
        state = frontier.pop()  # Updates the current state.
    # Test with the following command line prompt: python3 py/pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    # A* expands the cheapest node first while taking into consideration a Heuristic.
    # Check above for DFS, BFS, and UCS implementations
    # This method is essentially UCS with the inclusion of a heuristic when updating successor_cost
    closedset = []  # used to avoid cycle issues and infinite loops
    actions = []  # create a list of actions–this is what we are instructed to return
    legal_moves = problem.getStartState()  # first positioning
    ''' Next, we initialize the frontier using the provided util.py file, as the instructions hint at us to do
            UCS is implemented using a Priority Queue type structure or a Heap. In our case, we will be using a 
            PriorityQueue, which orders the nodes from cheapest to most expensive and expands them in that order
            '''
    expanded = PriorityQueue()  # This PriorityQueue is for already expanded states, while the next PriorityQueue is for
    # specifically for the fringe/frontier
    frontier = PriorityQueue()
    frontier.push(legal_moves, 0)  # Push first positioning plus a 0, which in this case represents the initial cost of
    # actions
    state = frontier.pop()  # represents current state of pacman

    while not problem.isGoalState(state):  # We are done, solution has been found if this is TRUE
        if not state in closedset:  # cycle checking
            closedset.append(state)
            # As before, next_state is a set of COORDINATES, action is a DIRECTION, cost is a NUMBER
            for next_state, action, cost in problem.getSuccessors(state):
                # How much the successor path costs, this time including the HEURISTIC. This line of code is the only
                # difference between A* and UCS (at last for my implementation)
                successor_cost = problem.getCostOfActions(actions + [action])  + heuristic(next_state, problem)
                if next_state not in closedset:  # more cycle checking
                    frontier.push(next_state, successor_cost)  # same idea as BFS/DFS
                    expanded.push(actions + [action], successor_cost)

        actions = expanded.pop()  # Updates the determined cheapest path.
        state = frontier.pop()  # Updates the current state.
    # Test with the following command line prompt:
    # python3 py/pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
