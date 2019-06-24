import read, copy
from util import *
from logical_classes import *

verbose = 0

class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        self.facts = facts
        self.rules = rules
        self.ie = InferenceEngine()

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def _get_fact(self, fact):
        """INTERNAL USE ONLY
        Get the fact in the KB that is the same as the fact argument

        Args:
            fact (Fact): Fact we're searching for

        Returns:
            Fact: matching fact
        """
        for kbfact in self.facts:
            if fact == kbfact:
                return kbfact

    def _get_rule(self, rule):
        """INTERNAL USE ONLY
        Get the rule in the KB that is the same as the rule argument

        Args:
            rule (Rule): Rule we're searching for

        Returns:
            Rule: matching rule
        """
        for kbrule in self.rules:
            if rule == kbrule:
                return kbrule

    def kb_add(self, fact_rule):
        """Add a fact or rule to the KB
        Args:
            fact_rule (Fact|Rule) - the fact or rule to be added
        Returns:
            None
        """
        printv("Adding {!r}", 1, verbose, [fact_rule])
        if isinstance(fact_rule, Fact):
            if fact_rule not in self.facts:
                self.facts.append(fact_rule)
                for rule in self.rules:
                    self.ie.fc_infer(fact_rule, rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.facts.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.facts[ind].supported_by.append(f)
                else:
                    ind = self.facts.index(fact_rule)
                    self.facts[ind].asserted = True
        elif isinstance(fact_rule, Rule):
            if fact_rule not in self.rules:
                self.rules.append(fact_rule)
                for fact in self.facts:
                    self.ie.fc_infer(fact, fact_rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.rules.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.rules[ind].supported_by.append(f)
                else:
                    ind = self.rules.index(fact_rule)
                    self.rules[ind].asserted = True

    def kb_assert(self, fact_rule):
        """Assert a fact or rule into the KB

        Args:
            fact_rule (Fact or Rule): Fact or Rule we're asserting
        """
        printv("Asserting {!r}", 0, verbose, [fact_rule])
        self.kb_add(fact_rule)

    def kb_ask(self, fact):
        """Ask if a fact is in the KB

        Args:
            fact (Fact) - Statement to be asked (will be converted into a Fact)

        Returns:
            listof Bindings|False - list of Bindings if result found, False otherwise
        """
        print("Asking {!r}".format(fact))
        if factq(fact):
            f = Fact(fact.statement)
            bindings_lst = ListOfBindings()
            # ask matched facts
            for fact in self.facts:
                binding = match(f.statement, fact.statement)
                if binding:
                    bindings_lst.add_bindings(binding, [fact])

            return bindings_lst if bindings_lst.list_of_bindings else []

        else:
            print("Invalid ask:", fact.statement)
            return []

    def kb_retract(self, fact_or_rule):
        """Retract a fact from the KB

        Args:
            fact (Fact) - Fact to be retracted

        Returns:
            None
        """
        printv("Retracting {!r}", 0, verbose, [fact_or_rule])
        ####################################################
        # Implementation goes here
        # Not required for the extra credit assignment

    def kb_explain(self, fact_or_rule):
        """
        Explain where the fact or rule comes from

        Args:
            fact_or_rule (Fact or Rule) - Fact or rule to be explained

        Returns:
            string explaining hierarchical support from other Facts and rules
        """
        ####################################################
        # Student code goes here
        explain = ""
        if isinstance(fact_or_rule, Fact):
            if fact_or_rule not in self.facts:
                explain = "Fact is not in the KB"
                return explain
            explain = explain + fact_or_rule.name + ": " +str(fact_or_rule.statement)
            index = self.facts.index(fact_or_rule)
            fact_or_ruleInKB = self.facts[index]
            #print(self.facts[self.facts.index(fact_or_rule)].supported_by)
            if fact_or_ruleInKB.supported_by != []:
                indent = 0
                #explain = explain + "\n  SUPPORTED BY"
                #print("heree")
                #print(explain)
                #print("heree end")
                explain = self.kb_explain_supported_by(fact_or_ruleInKB, indent, explain)
                """
                #supported by facts
                for i in fact_or_ruleInKB.supported_by:
                    indent += 1
                    print(i)
                    print()
                    explain = explain + "\n"
                    j = 0
                    while j < indent:
                        explain += "    "
                        j += 1
                    explain += i[0].name + ": " + str(i[0].statement)
                    if i[0].asserted:
                        explain += " ASSERTED"
                #supported by rules
                    explain = explain + "\n"
                    j = 0
                    while j < indent:
                        explain += "    "
                        j += 1
                    explain += i[1].name + ": (" + str(i[1].lhs[0])
                    for k in range(1, len(i[1].lhs)):
                        explain += ", " + str(i[1], lhs[k])
                    explain += ") -> " + str(i[1].rhs)
                    """

        else:
            if fact_or_rule not in self.rules:
                explain = "Rule is not in the KB"
                return explain
            explain = explain + fact_or_rule.name + ": " + str(fact_or_rule.statement)
            index = self.facts.index(fact_or_rule)
            fact_or_ruleInKB = self.facts[index]
            if fact_or_ruleInKB.supported_by != []:
                indent = 0
                explain = self.kb_explain_supported_by(fact_or_ruleInKB, indent, explain)



        print("explain")
        print(explain)
        print("explain end")
        print("answer")

        print('\
fact: (eats nyala leaves)\n\
  SUPPORTED BY\n\
    fact: (eats herbivore leaves) ASSERTED\n\
    rule: ((eats herbivore leaves)) -> (eats nyala leaves)\n\
      SUPPORTED BY\n\
        fact: (genls antelope herbivore) ASSERTED\n\
        rule: ((genls antelope ?z), (eats ?z leaves)) -> (eats nyala leaves)\n\
          SUPPORTED BY\n\
            fact: (genls nyala antelope) ASSERTED\n\
            rule: ((genls ?x ?y), (genls ?y ?z), (eats ?z leaves)) -> (eats ?x leaves) ASSERTED\n\
  SUPPORTED BY\n\
    fact: (isa leaves plantBasedFood) ASSERTED\n\
    rule: ((isa ?y plantBasedFood)) -> (eats nyala ?y)\n\
      SUPPORTED BY\n\
        fact: (eats nyala plantBasedFood) ASSERTED\n\
        rule: ((eats ?x plantBasedFood), (isa ?y plantBasedFood)) -> (eats ?x ?y) ASSERTED\
')
        print("answer end")
        return explain

    def kb_explain_supported_by(self, fact_or_ruleInKB, indent, explain):
        # supported by facts
        #print("enter")
        #explain = explain + "\n"
        #j = 0
        #while j < indent:
        #    explain += "    "
        #    j += 1
        #explain += "  SUPPORTED BY"
        for i in fact_or_ruleInKB.supported_by:
            explain = explain + "\n"
            j = 0
            while j < indent:
                explain += "    "
                j += 1
            explain += "  SUPPORTED BY"
            #print("wtf???")
            indent += 1
            #print(i)
            #print()
            explain = explain + "\n"
            j = 0
            while j < indent:
                explain += "    "
                j += 1
            explain += i[0].name + ": " + str(i[0].statement)
            if i[0].asserted:
                explain += " ASSERTED"
            else:
                explain = self.kb_explain_supported_by(i[0], indent, explain)
            # supported by rules
            explain = explain + "\n"
            j = 0
            while j < indent:
                explain += "    "
                j += 1
            #print("dddddddddddddddddddddddddddd")
            #print(i)
            #print(isinstance(i[0],Fact))
            #print(isinstance(i[1], Rule))
            explain += i[1].name + ": (" + str(i[1].lhs[0])
            for k in range(1, len(i[1].lhs)):
                explain += ", " + str(i[1].lhs[k])
            explain += ") -> " + str(i[1].rhs)
            if i[1].asserted:
                explain += " ASSERTED"
            else:
                #j = 0
                #explain += "\n"
                #while j < indent:
                #    explain += "    "
                #    j += 1
                #explain += " SUPPORTED BY"
                explain = self.kb_explain_supported_by(i[1], indent, explain)
            indent -= 1
        #print("IN")
        #print(explain)
        #print("IN end")
        return explain

class InferenceEngine(object):
    def fc_infer(self, fact, rule, kb):
        """Forward-chaining to infer new facts and rules

        Args:
            fact (Fact) - A fact from the KnowledgeBase
            rule (Rule) - A rule from the KnowledgeBase
            kb (KnowledgeBase) - A KnowledgeBase

        Returns:
            Nothing            
        """
        printv('Attempting to infer from {!r} and {!r} => {!r}', 1, verbose,
            [fact.statement, rule.lhs, rule.rhs])
        ####################################################
        # Implementation goes here
        # Not required for the extra credit assignment
