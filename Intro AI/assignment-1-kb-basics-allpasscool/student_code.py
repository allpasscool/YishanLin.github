import read, copy
from util import *
from logical_classes import *


class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        self.facts = facts
        self.rules = rules

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def kb_assert(self, fact):
        """Assert a fact or rule into the KB

        Args:
            fact (Fact or Rule): Fact or Rule we're asserting in the format produced by read.py
        """
        #print(fact)
        #print("123")
        #print(type(self.facts))
        #fact
        if "fact" in fact.name:

            #if true
            if fact.asserted:
                #print("Was asserted: ")
                #print(fact.asserted)
                if self.facts:
                    for i in self.facts:
                        if(i.statement == fact.statement):
                            print("already existed statment")
                            return
                self.facts.append(fact)
                return
            #if false
            else:
                print("A fact not asserted")
                return
        #rule
        if "rule" in fact.name:
            print("rule")
            return
        print("Asserting {!r}".format(fact))
        
    def kb_ask(self, fact):
        """Ask if a fact is in the KB

        Args:
            fact (Fact) - Fact to be asked

        Returns:
            ListOfBindings|False - ListOfBindings if result found, False otherwise
        """
        #match
        bindings = []
        for i in self.facts:
            tmpB = match(fact.statement, i.statement)
            if (tmpB):
                bindings.append(tmpB)

        if(bindings is []):
            return False
        print("Asking {!r}".format(fact))
        return bindings