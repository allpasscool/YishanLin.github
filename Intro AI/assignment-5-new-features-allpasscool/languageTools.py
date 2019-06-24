import spacy

rd = "\033[31;1m"
em = "\033[4m"
st = "\033[0m"

nlp = spacy.load('en')

# A set of terms that are indicative of different Statement types
# We want to be able to recognize four kinds of statements:
# 
# Existential statements of fact that define new objects: There are five apples on the table
# Feature statements of fact that refine features of existing objects: The apples are ripe
# 
# Commands that tell the system what needs to be done: Put the apple in the box
#
# Existential questions: Are there any red blocks on the table?
# Counting questions: How many red blocks are on the table?
# Feature questions: What color is the block on the table?
# 
# To determine the category, we are going to test the ROOT word of the sentence and words 
# that are linked to it
#
# We are going to use a simple pattern based tester.  Each pattern have three parts:
#   A check on the position of a core term
#   A specific child relationship we need
#   A test against its children
#   The Statement Type that is indicated

class statementRule:

    statementRules = []
    
    def __init__(self, testTerm, termPosition, childTest, childRelationship, category):
        self.testTerm = testTerm
        self.termPosition = termPosition
        self.childTest = childTest
        self.childRelationship = childRelationship
        self.category = category  
        self.weight = len(self.childTest) + len(self.childRelationship) + self.termPosition
        self.statementRules.append(self)
        
    def __str__(self):
        return "Category: " + self.category
        
# Statement rules

#statementRule("be",0,"","","S:Existential")
statementRule("be",0,"","acomp","S:Feature")
statementRule("be",0,"","nsubj","S:Feature")

# Command rules
statementRule("put",1,"","","Command")
statementRule("move",1,"","","Command")
statementRule("remove",1,"","","Command")
statementRule("place",1,"","","Command")

#Question rules

statementRule("be",1,"","","Q:Existential")
statementRule("be",1,"","acomp","Q:Feature")
statementRule("many",0,"how","advmod","Q:Count")
statementRule("be",0,"where","advmod","Q:Feature")
statementRule("what",1,"","","Q:Feature")

# We can categorize the sentences we have with the rules we have defined.
# Rules are weighted by how specific they are with more specific rules dominating

def catagorize(sentence):
    if sentence[0].lemma_ == "there": return "S:Existential"
    categories = []
    for rule in statementRule.statementRules:
        if rule.termPosition == 0:
            for token in sentence[1:]:
                if token.lemma_ == rule.testTerm and testRelationships(token.children, rule):
                    categories.append(rule)
                    break
        else:
            token = sentence[rule.termPosition-1]
            if rule.testTerm == token.lemma_ and testRelationships(token.children, rule):
                    categories.append(rule) 
    return sorted(categories, key=lambda rule: rule.weight, reverse=True)[0].category

def testRelationships(tokens, rule):
    if rule.childTest == "" and rule.childRelationship == "":
        return True
    else:
        for token in tokens:
            if token.lemma_ == rule.childTest and token.dep_ == rule.childRelationship:
                return True
            elif rule.childTest == "" and token.dep_ == rule.childRelationship:
                return True
            if token.lemma_ == rule.childTest and rule.childRelationship == "":
                return True
    return False

# A set of functions for extracting relationships
# We start with the root 

def extractRoot(document):
    for x in document:
        if x.dep_ == "ROOT":
            return x

# In those instances where there is a statement of fact, we need to know what
# the objects that are being talked about

def extractExistentialTarget(token):
    for x in token.children:
        if x.dep_ == "attr":
            return x


# In those instances where there is a statement of fact, we need to know what
# the objects that are being talked about

def extractSubject(token):
    for x in token.children:
        if x.dep_ == "nsubj":
            return x

def extractSubjectDeep(token):
    for x in token.children:
        if x.dep_ == "nsubj":
            if x.pos_ == "NOUN": return x
            else:
                deeper =moveThroughModifiers(x)
                if deeper: return deeper

def moveThroughModifiers(token):
    children = token.children
    if children == []: return False
    next = [a for a in children][0]
    if next.pos_ == "NOUN": return next
    elif next.pos_ in ["DET","ADP","PREP"]: return moveThroughModifiers(next)


# In those instances where there is a statement of new feature, we need to know what
# that feature is 

def extractTargetFeature(token):
    for x in token.children:
        if x.dep_ == "acomp":
            return x


# Given we know that there is an action, we need to extract the object of that action

def extractDirectObject(token):
    for x in token.children:
        if x.dep_ == "dobj" or x.dep_ == "punct" or x.dep_ == "attr":
            return x

# Any prepositions associated with the object (to, from, on, under)

def extractPreps(token):
    preps = []
    for x in token.children:
        if x.dep_ == "prep":
            preps.append(x)
    return preps

# Given we know that there is an action, we need to extract the object of that action

def extractPrepsAll(token):
    preps = []
    for x in token.children:
        if x.dep_ == "prep":
            preps.append(x)
        if x.dep_ == "dobj":
        	possibles = extractPreps(x)
        	if possibles:
        		preps = preps + possibles
    return preps


# The follow on is the object of the preposition

def extractObjectFromPrep(token):
    return [a for a in token.children][0]

# Given we know that there is an action, we need to extract the object of that action

def extractPrepObjects(token):
    prepObjs = []
    for x in token.children:
        if x.dep_ == "prep":
            prepObjs.append(extractObjectFromPrep(x))
    return prepObjs

# Given we know that there is an action, we need to extract the object of that action and all of the prep objects
# of its direct object

def extractPrepObjectsAll(token):
    prepObjs = []
    for x in token.children:
        if x.dep_ == "prep":
            prepObjs.append(extractObjectFromPrep(x))
        if x.dep_ == "dobj":
            possibles = extractPrepObjects(extractObjectFromPrep(x))
            if possibles:
                prepObjs = prepObjs + possibles
    return prepObjs

# Every object has a set of features associated with it that will define how we are going to
# deal with it.  We will to extract them separately
 
# Features that are direct modifiers (size, color, shape) 
           
def extractFeatures(token):
    features = []
    for x in token.children:
        if x.dep_ in ["amod","compound","acomp"]:
            features.append(x)
    return features


# Any determiner (a, an, the) associated with the object 

def extractDeterminer(token):
    for x in token.children:
        if x.dep_ == "det":
            return x



  
    
# The follow on is the object of the preposition 

def extractPrepObject(token):
    return [a for a in token.children][0]


# Any count associated with the object

def extractCount(token):
    for x in token.children:
        if x.dep_ == "nummod":
            return x


# Given a question that is looking for a feature, we need to get to what that feature is

def extractQuestionDetail(token):
    for x in token.children:
        if x.dep_ == "advmod":
            return x
            
            
# And we need to do a little bit of testing
#
# We need to know if an object that is being referred to is the name of an object, a 
# description of an object, or a reference to an object that is under discussion
# we do this by considering what the object is

def determineObjectType(token):
    if token.pos_ == "NOUN":
        return "noun"
    elif token.pos_ == "PRON":
        return "pronoun"
    elif token.pos_ == "NUM":
        if extractPreps(token):
            object = extractPrepObject(extractPreps(token)[0])
            if object.pos_ == "NOUN":
                return "plural noun"
            if object.pos_ == "PRON":
                return "plural pronoun"

# Utility functions for displaying examples
        
def showSet(examples):
    for sentence in examples:
        doc = nlp(sentence)
        print(rd+"\n****************************\n"+st)
        print(rd+"Sentence:"+st,sentence, "\n")
        
        for token in doc:
            showToken(token)

def showTree(doc):
    print(rd+"\n****************************\n"+st)
    for token in doc:
        showToken(token)


def showToken(token):
    if len(token.text) >= 7:
        tt =""
    else:
        tt="\t"
    print(rd+token.text+st+tt, "\t"+em+"STEM:"+st, token.lemma_, "\t"+em+"POS:"+st,token.pos_,
          "\t"+em+"REL:"+st,token.dep_, "\t"+em+"HEADER:"+st,token.head,
          "\t"+em+"Children:"+st,[x.lemma_ for x in token.children])
        
def trackReference(item):
    if item.pos_ == "NOUN":
        return item
    if item.pos_ == "NUM":
        return trackReference([a for a in item.children][0])
    elif item.pos_ == "ADP":
        return trackReference([a for a in item.children][0])

        
            
