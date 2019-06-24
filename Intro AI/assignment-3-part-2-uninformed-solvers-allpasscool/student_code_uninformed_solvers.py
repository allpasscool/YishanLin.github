
from solver import *

class SolverDFS(UninformedSolver):
    def __init__(self, gameMaster, victoryCondition):
        #super().__init__(gameMaster, victoryCondition)
        self.gm = gameMaster
        self.nextG = [GameState(self.gm.getGameState(), 0, None)]
        self.visited = dict()
        self.currentState = GameState(self.gm.getGameState(), 0, None)
        self.visited[self.currentState] = True
        self.victoryCondition = victoryCondition

    def solveOneStep(self):
        """
        Go to the next state that has not been explored. If a
        game state leads to more than one unexplored game states,
        explore in the order implied by the GameMaster.getMovables()
        function.
        If all game states reachable from a parent state has been explored,
        the next explored state should conform to the specifications of
        the Depth-First Search algorithm.

        Returns:
            True if the desired solution state is reached, False otherwise
        """
        ### Student code goes here
        # win?
        if self.currentState.state == self.victoryCondition:
            #print(self.currentState.state)
            return True


        # return to root
        while (self.currentState.parent):
            self.gm.reverseMove(self.currentState.requiredMovable)
            self.currentState = self.currentState.parent

        while len(self.nextG) > 0:
            currentGState = self.nextG.pop(0)

            # move from root to currentGState
            targetBack = []
            tmp = currentGState
            while tmp.parent:
                targetBack.append(tmp.requiredMovable)
                tmp = tmp.parent
            while len(targetBack) > 0:
                self.gm.makeMove(targetBack.pop())
            # we are in currentGState
            self.currentState = currentGState


            #win?
            if self.currentState.state == self.victoryCondition:
                #print(self.currentState.state)
                return True

            # if current state is not visited
            hasntVisited = True
            for i in self.visited:
                if i.state == currentGState.state:
                    hasntVisited = False
                    break
            if hasntVisited:
                self.visited[currentGState] = True
                movable = self.gm.getMovables()
                tmpnext = []
                for m in movable:
                    self.gm.makeMove(m)
                    tmp = GameState(self.gm.getGameState(), currentGState.depth + 1, m)
                    currentGState.children.append(tmp)
                    tmp.parent = currentGState
                    self.gm.reverseMove(m)
                    tmpnext.append(tmp)
                for k in self.nextG:
                    tmpnext.append(k)
                self.nextG = tmpnext
                #print("QQ")
                #print(self.currentState.state)
                return False

            # if not root and visited
            if not hasntVisited:
                if currentGState.parent:
                    # return to root node
                    while currentGState.parent:
                        self.gm.reverseMove(currentGState.requiredMovable)
                        currentGState = currentGState.parent
                    continue


            # if current visited before and is root
            movable = self.gm.getMovables()
            tmpnext = []
            for m in movable:
                self.gm.makeMove(m)
                tmp = GameState(self.gm.getGameState(), currentGState.depth + 1, m)
                currentGState.children.append(tmp)
                tmp.parent = currentGState
                self.gm.reverseMove(m)
                tmpnext.append(tmp)
            for k in self.nextG:
                tmpnext.append(k)
            self.nextG = tmpnext

            # return to root node
            while currentGState.parent:
                self.gm.reverseMove(currentGState.requiredMovable)
                currentGState = currentGState.parent

        return False


class SolverBFS(UninformedSolver):
    def __init__(self, gameMaster, victoryCondition):
        #super().__init__(gameMaster, victoryCondition)
        self.gm = gameMaster
        self.nextG = [GameState(self.gm.getGameState(), 0, None)]
        self.visited = dict()
        self.currentState = GameState(self.gm.getGameState(), 0, None)
        self.visited[self.currentState] = True
        self.victoryCondition = victoryCondition

    def solveOneStep(self):
        """
        Go to the next state that has not been explored. If a
        game state leads to more than one unexplored game states,
        explore in the order implied by the GameMaster.getMovables()
        function.
        If all game states reachable from a parent state has been explored,
        the next explored state should conform to the specifications of
        the Breadth-First Search algorithm.

        Returns:
            True if the desired solution state is reached, False otherwise
        """
        ### Student code goes here
        # win?
        if self.currentState.state == self.victoryCondition:
            #print(self.currentState.state)
            return True


        #return to root
        while(self.currentState.parent):
            self.gm.reverseMove(self.currentState.requiredMovable)
            self.currentState = self.currentState.parent

        while len(self.nextG) > 0:
            currentGState = self.nextG.pop(0)

            #move from root to currentGState
            targetBack = []
            tmp = currentGState
            while tmp.parent:
                targetBack.append(tmp.requiredMovable)
                tmp = tmp.parent
            while len(targetBack) > 0:
                self.gm.makeMove(targetBack.pop())
            #we are in currentGState
            self.currentState = currentGState

            #win?
            if self.currentState.state == self.victoryCondition:
                return True

            #if current state is not visited
            hasntVisited = True
            for i in self.visited:
                if i.state == currentGState.state:
                    hasntVisited = False
                    break
            if hasntVisited:
                self.visited[currentGState] = True
                movable = self.gm.getMovables()
                for m in movable:
                    self.gm.makeMove(m)
                    tmp = GameState(self.gm.getGameState(), currentGState.depth + 1, m)
                    currentGState.children.append(tmp)
                    tmp.parent = currentGState
                    self.gm.reverseMove(m)
                    self.nextG.append(tmp)
                return False

            # if not root and visited
            if not hasntVisited:
                if currentGState.parent:
                    # return to root node
                    while currentGState.parent:
                        self.gm.reverseMove(currentGState.requiredMovable)
                        currentGState = currentGState.parent
                    continue

            #if current visited before
            movable = self.gm.getMovables()
            for m in movable:
                self.gm.makeMove(m)
                tmp = GameState(self.gm.getGameState(), currentGState.depth + 1, m)
                currentGState.children.append(tmp)
                tmp.parent = currentGState
                self.gm.reverseMove(m)
                self.nextG.append(tmp)

            #return to root node
            while currentGState.parent:
                self.gm.reverseMove(currentGState.requiredMovable)
                currentGState = currentGState.parent

        return False
