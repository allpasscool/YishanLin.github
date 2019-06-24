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
                    #print("\ntest")
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
                    #print("\ntest1")
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
        # Student code goes here
        #if isinstance(fact_or_rule, Rule):
        #    print("Shitttttttttttttttttttt")
        #    return

        if fact_or_rule in self.facts:
            #if fact_or_rule.asserted:
            indexOfF = self.facts.index(fact_or_rule)
            #is supported by others
            if self.facts[indexOfF].supported_by:
                fact_or_rule.asserted = False
                return
            #is not supported
            else:
                for i in self.facts[indexOfF].supports_facts:
                    for j in i.supported_by:
                        if self.facts[indexOfF] in j:
                            i.supported_by.remove(j)
                    self.kb_delFact(i)
                for i in self.facts[indexOfF].supports_rules:
                    for j in i.supported_by:
                        if self.facts[indexOfF] in j:
                            i.supported_by.remove(j)
                    self.kb_delRule(i)
                self.facts.remove(fact_or_rule)

    def kb_delFact(self, fact):
        #if not fact.supported_by:
        if len(fact.supported_by) == 0:
            self.facts.remove(fact)
            for i in fact.supports_facts:
                for j in i.supported_by:
                    if fact in j:
                        i.supported_by.remove(j)
                self.kb_delFact(i)


    def kb_delRule(self, rule):
        if not rule.supported_by:
            self.rules.remove(rule)
            for i in rule.supports_facts:
                for j in i.supported_by:
                    if rule in j:
                        i.supported_by.remove(j)
                self.kb_delRule(i)

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
        # Student code goes here
        #Use the util.match function to do unification and create possible bindings
        #Use the util.instantiate function to bind a variable in the rest of a rule
        #Rules and Facts have fields for supported_by, supports_facts, and supports_rules. Use them to track inferences! For example, imagine that a fact F and rule R matched to infer a new fact/rule fr.
        #fr is supported by F and R. Add them to fr's supported_by list - you can do this by passing them as a constructor argument when creating fr.
        #F and R now support fr. Add fr to the supports_rules and supports_facts lists (as appropriate) in F and R.

        if not rule.lhs:
            print("not rule.lhs")
            #return

        bind = match(fact.statement, rule.lhs[0])
        if bind:
            inferS = instantiate(rule.rhs, bind)
            indexOfF = kb.facts.index(fact)
            indexOfR = kb.rules.index(rule)
            lenOfFacts = len(kb.facts)
            lenOfRules = len(kb.rules)

            """if fact in kb.facts:
                print("find F")
                print(fact == kb.facts[0])
            else:
                print("didn't find F")
            """
            """
            print("indexOfF")
            print(indexOfF)
            print("lenOfFacts")
            print(lenOfFacts)
            print("indexOfR")
            print(indexOfR)
            print("lenOfRules")
            print(lenOfRules)
            """
            #if indexOfF == lenOfFacts - 1:
            if len(rule.lhs) == 1:
                #print("new fact not in kb")
                inferF = Fact(inferS, [[fact, rule]])
                #inferF = Fact(inferS)
                #fact.supports_facts.append(inferF)
                #rule.supports_facts.append(inferF)
                if inferF in kb.facts:
                    print("WTF???????????????????????????????????????????????????????")
                if inferF not in kb.facts:
                    #inferF.supported_by.append([fact, rule])
                    fact.supports_facts.append(inferF)
                    rule.supports_facts.append(inferF)
                    kb.kb_assert(inferF)
            #elif indexOfR == lenOfRules - 1:
            else:
                #print("new rule not in kb")
                tmp = []
                for i in rule.lhs[1:]:
                    tmp.append(instantiate(i, bind))
                inferR = Rule([tmp, inferS], [[fact, rule]])
                #inferR = Rule([tmp, inferS])

                #fact.supports_rules.append(inferR)
                #rule.supports_rules.append(inferR)
                if inferR in kb.rules:
                    print("OKwtf???????????????????????????????????????????????????????????")
                if inferR not in kb.rules:
                    #inferR.supported_by.append([fact, rule])
                    fact.supports_rules.append(inferR)
                    rule.supports_rules.append(inferR)
                    kb.kb_add(inferR)
        #print("term : ")
        #print(term)
        #print("\n")