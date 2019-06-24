
global KB
KB = []

global RB
RB = []

class statement(object):
    
    # A statement has a predicate, a set of arguments that is applies to and a list of
    # statements/rules that it supports (initially empty). Arguments are processed to
    # turn both variables and constants into objects.
    
    def __init__(self, pattern):
        self.full = pattern
        self.predicate = pattern[0].upper()
        self.args = pattern[1:]
        self.facts = []
        self.rules = []
    
    
    # Pretty hands back a nice looking string version of this object for printing
    
    def pretty(self):
        return "(" + " ".join(self.full) + ")"
    
    
    # Add inferred facts and rules to a statement so that we can track them for later retraction
    
    def add_fact(self, fact):
        for f in self.facts:
            if f.full == fact.full:
                return
        self.facts.append(fact)

    def add_rule(self, rule):
        for r in self.rules:
            if r.pretty == rule.pretty:
                return
        self.rules.append(rule)


# Rules have a few more elements to them.

# They have:
#  A left hand side (a list of patterns that have to be true for a rule to fire).
#  A right hand side (a pattern that needs to be instantiated)
#  A name (for convenience sake and use later).
#  The same list of facts and rules that it supports that facts have
#  The type of rule (does it Assert or Retract its conclusion
#  Note: Any rule where the right hand side begins with a "~" is marked as a "Retract"

class rule(object):
    
    count = 0
    
    def __init__(self, lhs, rhs):
        self.full = (lhs, rhs)
        self.type = "Assert"
        self.name = "Rule "+ str(rule.count)
        self.lhs = [a for a in map(lambda x: statement(x), lhs)]
        if rhs[0][0] == "~":
            rhs[0] = rhs[0][1:]
            self.type = "Retract"
        self.rhs = statement(rhs)
        self.facts = []
        self.rules = []
        rule.count = rule.count + 1

    
    # Pretty hands back a nice looking string version of this object for printing
    
    def pretty(self):
        return self.name + ": When <"+ " ".join([a for a in map(lambda x: x.pretty(), self.lhs)]) + "> " + self.type + " " + self.rhs.pretty()
    
    
    # Add inferred facts and rules to a statement so that we can track them for later retraction

    def add_fact(self, fact):
        for f in self.facts:
            if f.full == fact.full:
                return
        self.facts.append(fact)

    def add_rule(self, rule):
        for r in self.rules:
            if r.pretty == rule.pretty:
                return
        self.rules.append(rule)


# Match is designed around the statement structure

# Match takes two arguments, a pattern and a fact and returns either a bindings list
# if they match or False if they do not.

# It checks to see if the predicates are the same and then tests the arguments against
# each other.

# The pattern may have variables in it. The fact may not.

def match (pattern,fact):
    p = pattern.full
    f = fact.full
    if p[0] != f[0]:
        return False
    return match_args(p[1:],f[1:])


# The pattern may have variables in it. The fact may not.

def pattern_match (pattern,fact):
    p = pattern
    f = fact.full
    if p[0] != f[0]:
        return False
    return match_args(p[1:],f[1:])


# Match args just steps through two lists of arguments and returns the list of bindings
# that have to be in place if they are going to match.  If it ever finds a mismatch,
# it returns False and stops checking.

# It maintains a set of bindings so that repeated use of a variable can be noted and enforced.

def match_args(pattern_args, fact_args):
    bindings = {}
    for p,f in zip(pattern_args, fact_args):
        bindings = match_element(p, f, bindings)
        if False == bindings:
            return False
    return bindings


# Match_element takes two elements and a list of bindings and returns the bindings under
# which the elements match.  If both elements are contants, then it just checks to see if
# they are the same.
# If the first element is a variable, it checks to see if it is already bound and tests the
# other element against the value.
# If the variable is unbound, it adds the binding to the bindings list and returns it.

def match_element(p, f, bindings):
    if p == f:
        return bindings
    elif varq(p):
        bound = bindings.get(p, False)
        if bound:
            if f == bound:
                return bindings
            else:
                return False
        else:
            bindings[p] = f
            return bindings
    else:
        return False


# Instantiate takes a pattern and a set of bindings and creates a new statement with the
# variables replaced with the values in the bindings list. It walks through the arguments
# and replaces any element that is in the bindings list with its value

def instantiate(pattern, bindings):
    predicate = pattern[0]
    args = [a for a in map(lambda x: bindings.get(x, x), pattern[1:])]
    args.insert(0, predicate)
    return args


# varq just checks to see if an element is a variable

def varq(item):
    if item[0] == "?":
        return True
    else:
        return False


# Assert takes a statement, checks to see if it is on the list of facts and adds it to 
# the KB list.  It then calls Infer_from_fact with the new fact.

def Assert(fact, show=False):
    if show:
        print("Asserting fact: " + fact.pretty())
    for f in KB:
        if f.full == fact.full:
            if show:
                print("\t" + fact.pretty(), "is already in KB")
            return
    KB.append(fact)
    Infer_from_fact(fact)


# Assert_Rule takes a rule, checks to see if it is on the list of rules and adds it to 
# the RB list.  It then calls Infer_from_rule with the new rule.

def Assert_Rule(new_rule, show=False):
    if show:
        print("Asserting rule:", new_rule.pretty())
    for r in RB:
        if r.full == new_rule.full:
            if show:
                print("\t",r.pretty(),"is already in Rule Base")
            return
    RB.append(new_rule)
    Infer_from_rule(new_rule)
    

# Infer_from_fact looks through all of the rules on the RB list.  If the fact matches the 
# first element of the lhs (the test) of a rule, then either the rule has a single test
# and the conclusion is instantiated or there are still tests to be checked and the entire 
# rule (with the one element of the test removed) is instantiated with the bindings.
# 
# If a fact is inferred, we Assert it.  If a new rule is built, we Assert_Rule it.
  
def Infer_from_fact(fact):
#    print("\tChecking fact:", fact.pretty())
    for r in RB:
        bindings = match(r.lhs[0], fact)
        if bindings != False:
            if len(r.lhs) == 1:
                new_statement = statement(instantiate(r.rhs.full, bindings))
                fact.add_fact(new_statement)
                r.add_fact(new_statement)
                if r.type == "Assert":
#                    print("\tInfering new fact from fact:", new_statement.pretty())
#                    print("\t\t" + r.pretty(), "\n\t\tFact:", fact.pretty())
                    Assert(new_statement)
                else:
                    print("\tRetracting:", new_statement.pretty())
            else:
                tests = [a for a in map(lambda x: instantiate(x.full, bindings), r.lhs[1:])]
                rhs = instantiate(r.rhs.full, bindings)
                new_rule = rule(tests, rhs)
                new_rule.type = r.type
                fact.add_rule(new_rule)
                r.add_rule(new_rule)
                print("\tInfering refined rule:", new_rule.pretty())
                Assert_Rule(new_rule, False)

# Infer_from_rule is exactly like Infer_from_fact except that it looks through all of the  
# facts on the KB list.  For each fact, if matches the first element of the lhs (the test) 
# of a rule, then either the rule has a single test
# and the conclusion is instantiated or there are still tests to be checked and the entire 
# rule (with the one element of the test removed) is instantiated with the bindings.
# 
# If a fact is inferred, we Assert it.  If a new rule is built, we Assert_Rule it. 
 
def Infer_from_rule(r):
#    print("\tChecking rule:", rule.pretty())
    for f in KB:
        bindings = match(r.lhs[0], f)
        if bindings != False:
            if len(r.lhs) == 1:
                new_statement = statement(instantiate(r.rhs.full, bindings))
                r.add_fact(new_statement)
                f.add_fact(new_statement)
                if r.type == "Assert":
                    Assert(new_statement)
                else:
                    print("\tRetracting:", new_statement.full)
            else:
                lhs = [a for a in map(lambda x: instantiate(x.full, bindings), r.lhs[1:])]
                rhs = instantiate(r.rhs.full, bindings)
                new_rule = rule(lhs, rhs)
                new_rule.type = r.type
                r.add_rule(new_rule)
                f.add_rule(new_rule)
                Assert_Rule(new_rule, False)


            
def remove_fact_and_supports(fact):
    if fact in KB:
        print("Removing:", fact.pretty())
        for f in fact.facts:
            remove_fact_and_supports(f)
        KB.remove(fact)

def Ask(pattern):
    list_of_bindings_lists = []
    for fact in KB:
        bindings = pattern_match(pattern, fact)
        if bindings != False:
            list_of_bindings_lists.append(bindings)
#    for b in list_of_bindings_lists:
#        print("This is true: \t",
#               statement(instantiate(pattern, b)).pretty())
    return list_of_bindings_lists
        
def Ask(queries):
    bindings_lists = []
    for query in queries:
        if bindings_lists != []:
            # So I already have matched something, so onto the next
            # For evey one of the elements I have matched, I want to see if I can match more
            # if I don't match anything for any ONE of these, I want to lose that binding
            for pair in [a for a in map(lambda b: (instantiate(query, b), b), bindings_lists)]:
                match = False
                new_bindings_list = []
                for fact in KB:
                    new_bindings = pattern_match(pair[0], fact)
                    print(fact.full)
                    print(pair[0])
                    print(new_bindings)
                    if new_bindings != False:
                        for key in pair[1]:
                            new_bindings[key] = pair[1][key]
                            new_bindings_list.append(new_bindings)
            if new_bindings_list == []:
                return False
            else:
                bindings_list = new_bindings_list
        else:
            for fact in KB:
                new_bindings = pattern_match(query, fact)
                if new_bindings != False:
                    bindings_lists.append(new_bindings)
                    match = True
        if match == False:
            return False
    return bindings_lists


def Retract(statement):
    for fact in KB:
        if pattern_match(statement, fact) != False:
            remove_fact_and_supports(fact)


def betterAsk(queries, bindings_list = list()):
    if queries == []:
        return bindings_list
    
    new_bindings_list = []
    if bindings_list != []:
        for bindings in bindings_list:
            query = instantiate(queries[0], bindings)
            for fact in KB:
                new_bindings = pattern_match(query, fact)
                if new_bindings != False:
                    for key in bindings.keys():
                        new_bindings[key] = bindings[key]
                    new_bindings_list.append(new_bindings)
    else:
        query = queries[0]
        new_bindings_list = []
        for fact in KB:
            new_bindings = pattern_match(query, fact)
            if new_bindings != False:
                new_bindings_list.append(new_bindings)
    if new_bindings_list != []:
        return betterAsk(queries[1:],new_bindings_list)
    else:
        return False


def whatAmI(term):
    bindings_lists = betterAsk([["isa", term, "?x"]])
    if bindings_lists == False:
        return False
    else:
        return bindings_lists[0]["?x"]

def Category(term):
    return betterAsk([["class", term]])

def howMany(query):
    return betterAsk(query)

def assertFromStatement(sentence):
    Assert(statement(sentence))
