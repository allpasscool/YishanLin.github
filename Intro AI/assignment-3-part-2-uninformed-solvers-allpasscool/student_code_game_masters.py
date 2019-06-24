from game_master import GameMaster
from read import *
from util import *

class TowerOfHanoiGame(GameMaster):

    def __init__(self):
        super().__init__()
        
    def produceMovableQuery(self):
        """
        See overridden parent class method for more information.

        Returns:
             A Fact object that could be used to query the currently available moves
        """
        return parse_input('fact: (movable ?disk ?init ?target)')

    def getGameState(self):
        """
        Returns a representation of the game in the current state.
        The output should be a Tuple of three Tuples. Each inner tuple should
        represent a peg, and its content the disks on the peg. Disks
        should be represented by integers, with the smallest disk
        represented by 1, and the second smallest 2, etc.

        Within each inner Tuple, the integers should be sorted in ascending order,
        indicating the smallest disk stacked on top of the larger ones.

        For example, the output should adopt the following format:
        ((1,2,5),(),(3, 4))

        Returns:
            A Tuple of Tuples that represent the game state
        """
        ### student code goes here
        peg1 = list()
        peg2 = list()
        peg3 = list()
        #if(self.kb.facts):
        #    print("HI")
        for i in self.kb.facts:
            #print(str(i.statement))
            #print(str(i.statement.terms[0]))
            if "on" in str(i.statement):
                tmp = 0
                #print(i.statement)
                for k in str(i.statement.terms[0]):
                    for j in k:
                        if j.isdigit():
                            tmp += int(j)
                if "peg1" in str(i.statement):
                    peg1.append(tmp)
                elif "peg2" in str(i.statement):
                    peg2.append(tmp)
                elif "peg3" in str(i.statement):
                    peg3.append(tmp)
        #print("peg1")
        #print(peg1)
        #print(peg2)
        peg1.sort()
        peg2.sort()
        peg3.sort()
        #print(peg1)
        peg1 = tuple(peg1)
        peg2 = tuple(peg2)
        peg3 = tuple(peg3)
        state = list()
        state.append(peg1)
        state.append(peg2)
        state.append(peg3)
        state = tuple(state)
        #print("state")
        #print(state)
        return state

    def makeMove(self, movable_statement):
        """
        Takes a MOVABLE statement and makes the corresponding move. This will
        result in a change of the game state, and therefore requires updating
        the KB in the Game Master.

        The statement should come directly from the result of the MOVABLE query
        issued to the KB, in the following format:
        (movable disk1 peg1 peg3)

        Args:
            movable_statement: A Statement object that contains one of the currently viable moves

        Returns:
            None
        """
        ### Student code goes here
        #print("movable_statement")
        #print(movable_statement)
        pred = movable_statement.predicate
        if pred not in "movable":
            print("wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        sl = movable_statement.terms
        #print(sl)
        disk = sl[0]
        pegFrom = sl[1]
        pegTo = sl[2]
        #print(type(disk))
        #print(type(pegTo))
        #print([disk, pegTo])
        #statement = Statement([disk, pegTo])
        #print("statement")
        #print(statement)
        #print(parse_input("fact: (on " + str(disk) + " " + str(pegFrom) + ")").statement)

        #retract old fact: on disk pegFrom
        oldFact = None
        found = False
        for i in self.kb.facts:
            if "on" in str(i.statement):
                if str(disk) in str(i.statement):
                    if str(pegFrom) in str(i.statement):
                        oldFact = i
                        found = True
                        break
        #print("found ?")
        #print(found)
        if not found:
            print("not found!!!!!!!!!!!!")
            return
        self.kb.kb_retract(oldFact)

        #retract old fact: top disk pegFrom
        for i in self.kb.facts:
            if "top" in str(i.statement):
                if str(disk) in str(i.statement):
                    if str(pegFrom) in str(i.statement):
                        oldFact = i
                        break
        #print("no found old fact?????")
        #print(oldFact)
        self.kb.kb_retract(oldFact)

        #retract of fact: top disk3 pegTo
        disk3 = False
        for i in self.kb.facts:
            if "top" in str(i.statement):
                if str(pegTo) in str(i.statement):
                    oldFact = i
                    disk3 = i.statement.terms[0]
                    self.kb.kb_retract(oldFact)
                    break
            if "empty" in str(i.statement):
                if str(pegTo) in str(i.statement):
                    oldFact = i
                    self.kb.kb_retract(oldFact)
                    break

        #retract ontop disk disk2
        disk2 = disk
        for i in self.kb.facts:
            if "ontop" in str(i.statement):
                if str(disk) in str(i.statement):
                    oldFact = i
                    #print("WTF")
                    #print(i.statement.terms)
                    disk2 = i.statement.terms[1]
                    self.kb.kb_retract(oldFact)
                    break

        #retract on disk pegFrom
        for i in self.kb.facts:
            if "on" in str(i.statement):
                if str(disk) in str(i.statement):
                    if str(pegFrom) in str(i.statement):
                        oldFact = i
                        break
        self.kb.kb_retract(oldFact)

        #add new fact: on disk pegTo
        newFact = parse_input("fact: (on " + str(disk) + " " + str(pegTo) + ")")
        self.kb.kb_assert(newFact)

        #add new fact: top disk PegTo
        newFact1 = parse_input("fact: (top " + str(disk) + " " + str(pegTo) + ")")
        self.kb.kb_assert(newFact1)

        if disk != disk2:
            #add new fact: top disk2 pegFrom
            newFact1 = parse_input("fact: (top " + str(disk2) + " " + str(pegFrom) + ")")
            self.kb.kb_assert(newFact1)
        else:
            #add new fact: empty pegFrom
            newFact1 = parse_input("fact: (empty " + str(pegFrom) + ")")
            self.kb.kb_assert(newFact1)

        #add new fact: ontop disk disk3
        if disk3:
            newFact1 = parse_input("fact: (ontop " + str(disk) + " " + str(disk3) + ")")
            self.kb.kb_assert(newFact1)
        #print("finish make move")
        #print(movable_statement)
        #for i in self.kb.facts:
        #    print(i)
        #print("finish make move 1_1")
        pass

    def reverseMove(self, movable_statement):
        """
        See overridden parent class method for more information.

        Args:
            movable_statement: A Statement object that contains one of the previously viable moves

        Returns:
            None
        """
        pred = movable_statement.predicate
        sl = movable_statement.terms
        newList = [pred, sl[0], sl[2], sl[1]]
        #print("reverse")
        #print(newList)
        self.makeMove(Statement(newList))

class Puzzle8Game(GameMaster):

    def __init__(self):
        super().__init__()

    def produceMovableQuery(self):
        """
        Create the Fact object that could be used to query
        the KB of the presently available moves. This function
        is called once per game.

        Returns:
             A Fact object that could be used to query the currently available moves
        """
        return parse_input('fact: (movable ?piece ?initX ?initY ?targetX ?targetY)')

    def getGameState(self):
        """
        Returns a representation of the the game board in the current state.
        The output should be a Tuple of Three Tuples. Each inner tuple should
        represent a row of tiles on the board. Each tile should be represented
        with an integer; the empty space should be represented with -1.

        For example, the output should adopt the following format:
        ((1, 2, 3), (4, 5, 6), (7, 8, -1))

        Returns:
            A Tuple of Tuples that represent the game state
        """
        ### Student code goes here
        row1 = list()
        row2 = list()
        row3 = list()

        tmpfacts = self.kb.facts
        for i in range(0, len(tmpfacts)):
            for j in range(i+1, len(tmpfacts)):
                if "pos" in str(tmpfacts[i].statement.predicate):
                    if "pos" in str(tmpfacts[j].statement.predicate):
                        if (str(tmpfacts[i].statement.terms[1]) > str(tmpfacts[j].statement.terms[1])):
                            tmp = tmpfacts[i]
                            tmpfacts[i] = tmpfacts[j]
                            tmpfacts[j] = tmp
                    if "empty" in str(tmpfacts[j].statement.predicate):
                        if (str(tmpfacts[i].statement.terms[1] )> str(tmpfacts[j].statement.terms[0])):
                            tmp = tmpfacts[i]
                            tmpfacts[i] = tmpfacts[j]
                            tmpfacts[j] = tmp
                if "empty" in str(tmpfacts[i].statement.predicate):
                    if "pos" in str(tmpfacts[j].statement.predicate):
                        if (str(tmpfacts[i].statement.terms[0]) > str(tmpfacts[j].statement.terms[1])):
                            tmp = tmpfacts[i]
                            tmpfacts[i] = tmpfacts[j]
                            tmpfacts[j] = tmp
                    if "empty" in str(tmpfacts[j].statement.predicate):
                        if (str(tmpfacts[i].statement.terms[0]) > str(tmpfacts[j].statement.terms[0])):
                            tmp = tmpfacts[i]
                            tmpfacts[i] = tmpfacts[j]
                            tmpfacts[j] = tmp
        for i in tmpfacts:
            if "pos" in str(i.statement.predicate):
                tmp = 0
                for k in str(i.statement.terms[0]):
                    for j in k:
                        if j.isdigit():
                            tmp += int(j)
                if "pos1" in str(i.statement.terms[2]):
                    row1.append(tmp)
                elif "pos2" in str(i.statement.terms[2]):
                    row2.append(tmp)
                elif "pos3" in str(i.statement.terms[2]):
                    row3.append(tmp)
            if "empty" in str(i.statement.predicate):
                tmp = -1
                if "pos1" in str(i.statement.terms[1]):
                    row1.append(tmp)
                elif "pos2" in str(i.statement.terms[1]):
                    row2.append(tmp)
                elif "pos3" in str(i.statement.terms[1]):
                    row3.append(tmp)

        #row1.sort()
        #row2.sort()
        #row3.sort()

        row1 = tuple(row1)
        row2 = tuple(row2)
        row3 = tuple(row3)
        state = list()
        state.append(row1)
        state.append(row2)
        state.append(row3)
        state = tuple(state)

        #print()
        #print(state)
        return state

    def makeMove(self, movable_statement):
        """
        Takes a MOVABLE statement and makes the corresponding move. This will
        result in a change of the game state, and therefore requires updating
        the KB in the Game Master.

        The statement should come directly from the result of the MOVABLE query
        issued to the KB, in the following format:
        (movable tile3 pos1 pos3 pos2 pos3)

        Args:
            movable_statement: A Statement object that contains one of the currently viable moves

        Returns:
            None
        """
        ### Student code goes here
        pred = movable_statement.predicate
        if pred not in "movable":
            print("wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        s = movable_statement.terms
        tile = s[0]
        posXFrom = s[1]
        posYFrom = s[2]
        posXTo = s[3]
        posYTo = s[4]

        # retract old fact: pos tile posXFrom posYFrom
        oldFact = None
        found = False
        for i in self.kb.facts:
            if "pos" in str(i.statement.predicate):
                if str(tile) in str(i.statement):
                    if str(posXFrom) in str(i.statement.terms[1]):
                        if str(posYFrom) in str(i.statement.terms[2]):
                            oldFact = i
                            found = True
                            break
        # print("found ?")
        # print(found)
        if not found:
            print("not found!!!!!!!!!!!!")
            return
        self.kb.kb_retract(oldFact)

        # retract old fact: empty posXTo posYTo
        for i in self.kb.facts:
            if "empty" in str(i.statement):
                if str(posXTo) in str(i.statement.terms[0]):
                    if str(posYTo) in str(i.statement.terms[1]):
                        oldFact = i
                        break
        # print("no found old fact?????")
        # print(oldFact)
        self.kb.kb_retract(oldFact)

        # add new fact: pos tile posXTo posYTo
        newFact = parse_input("fact: (pos " + str(tile) + " " + str(posXTo) + " " + str(posYTo) +")")
        self.kb.kb_assert(newFact)

        # add new fact: empty posXFrom posYFrom
        newFact1 = parse_input("fact: (empty " + str(posXFrom) + " " + str(posYFrom) + ")")
        self.kb.kb_assert(newFact1)

        pass

    def reverseMove(self, movable_statement):
        """
        See overridden parent class method for more information.

        Args:
            movable_statement: A Statement object that contains one of the previously viable moves

        Returns:
            None
        """
        pred = movable_statement.predicate
        sl = movable_statement.terms
        newList = [pred, sl[0], sl[3], sl[4], sl[1], sl[2]]
        self.makeMove(Statement(newList))
