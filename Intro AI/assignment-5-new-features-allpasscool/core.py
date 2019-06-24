import languageTools, utilities, reader, kb, memory, student_code

global longTermMemory
global statements
longTermMemory = memory.memory()
statements = []

#Red = "\u001b[31m"
#Grn = "\u001b[32m"
#Ylw = "\u001b[33m"
#Blu = "\u001b[34m"
#Mag = "\u001b[35m"
#Cyn = "\u001b[36m"
#
#rd = "\033[31;1m"
#em = "\033[4m"
#st = "\033[0m"

Red = ""
Grn = ""
Ylw = ""
Blu = ""
Mag = ""
Cyn = ""

rd = ""
em = ""
st = ""
lf = "\n"


# buildFOPC takes a sentence and Asseerts the facts that are derived from it
# It begins by using the language tools to build a syntactic parse and then,
# depending on the type of sentence, has it processed as a statement, question or command

def buildFOPC(sentence):

    global statements
    global recentMemory
    global longTermMemory

    statements.append(Red+sentence+st)

    recentMemory=memory.memory()

    print(lf+"Processing sentence:",sentence, lf)
    doc = languageTools.nlp(sentence)

    languageTools.showTree(doc)

    category = languageTools.catagorize(doc)
    print(lf+"Sentence is an instance of:", category)

    if category == "S:Existential":
        buildExistential(doc)
    if category == "S:Feature":
        student_code.buildFeatureStatement(doc)

    longTermMemory = recentMemory
    print(lf+"Long Term Memory", longTermMemory)


# buildExistential takes a parse tree and pulls out the elements necessary to start Asserting
# statements

def buildExistential(parse):

    # We begin with the ROOT (the verb) and the object we are defining. Given the object, we want to
    # figure out what to call it (resolve the reference).  We do this using resolveObjectFOPC, that
    # detemines what status the objec has and returns a set of names.  For some objects, resolveObjectFOPC
    # might Assert some facts about the object. Because the reference may be to multiple objects, it
    # returns a list.

    root = languageTools.extractRoot(parse)

    print(lf+Red+"ROOT:"+st, root)

    primary = languageTools.extractExistentialTarget(root)
    print(Red+"Primary Object:"+st, primary)

    names = resolveObjectFOPC(primary)

    # There may be modifiers to the object that are prepositions linked to the verb.
    # So we need to pull those out and, if they exist, process them as objects as well
    findAndAttachPrepObjectsFOPC(root,primary,names)


# resolveObject has to decide if the object description it is looking at is an existing object
# that is being descibed, an existing object that is referred to by name, or the description of
# a new object.

def resolveObjectFOPC(object):

    global recentMemory

    # First, we want to find out if the object is the name for anything we know about
    # To do this, we need to find out if it is a proper noun and if it exists in the KB.
    # If it does, then we are just going to return its name.

    print(lf+Grn+"Resolving object:"+st, object.lemma_)

    if object.pos_ == "PROPN":
        print(object.text, Grn+" is a proper name."+st)
        inst = kb.betterAsk(["inst", object.text,"?x"])
        recentMemory.addElements(inst,[object.text])
        print(Grn+"Returning"+st, object.text)
        return [object.text]

    # The object might also be a referent that quantifies something else. When we identify that we
    # then have to go find it.

    if object.pos_ == "ADJ":
        objects = discoverReferencedObjects(object)
        print(Grn+"Returning"+st, objects)
        return objects

    # Next we need to do is find out the determiner. If it is 'the', we are referring
    # an existing object and need to look it up.

    determiner = languageTools.extractDeterminer(object)
    if determiner != None:
        if determiner.lemma_ == "the":
            new_obj = discoverObject(object)
            print(Grn+"Returning"+st, new_obj)
            return new_obj

# The object might also be a referent that quantifies something else. When we identify that we
    # then have to go find it.

    if object.pos_ == "NUM":
        new_objs = discoverQuantifiedObjects(object)
        print(Grn+"Returning"+st, new_objs)
        return new_objs

    # If it is neither an existing object in the KB nor the name of a thing, then it is something
    # we have to build and assert

    new_objs = buildNewObject(object)
    print(Grn+"Returning"+st, new_objs)
    return new_objs


# buildNewObject create a description of an obejct (or objects) from the language and then Asserts
# It returns a list of the objects that it has built

def buildNewObject(object):

    print(Blu+"Referenced object is being defined in sentence"+st)
    print(Blu+"Building new object:"+st, object.lemma_)

    # Is this a statement about a single object or a set of objects
    # to do this to need to get the deteminer and the count we are only considering between
    # between 1 and 10. We also have a utility that converts names to numbers.

    determiner = languageTools.extractDeterminer(object)
    count = languageTools.extractCount(object)
    count_number = utilities.countNumber(determiner,count)
    print(Blu+"Count:"+st, str(count_number))

    # We generate new names for each of our new objects
    names = utilities.genNames(object.lemma_,count_number)
    print(Blu+"New Names:"+st, names)

    # Now we can begin making statements about our objects, first, we need to Assert that they exist
    buildAndAssertObjectStatementFOPC(object, names)

    # Next we pull out any features associated with the object and attach then to the object
    findAndAssertFeaturesFOPC(object,names)

    # And build up statement about any prepositional phrases linked to the object
    findAndAttachPrepObjectsFOPC(object,object,names)

    print(Blu+"Done building:"+st, object.lemma_)
    recentMemory.addElements(object.lemma_,names)
    return names

# discoverObject builds out a FOPC desription of an object with its "name" as a variable and then
# uses "betterAsk" to find a binding that matches those facts. It uses the features in the text to
# build these statements

def discoverObject(object):

    print(Mag+"Building concept to search KB for"+st,object.lemma_)

    features = languageTools.extractFeatures(object)
    facts = [["inst","?x",object.lemma_]]
    for f in features:
        type = kb.whatAmI(f.lemma_)
        facts.append([type,"?x",f.lemma_])
    print(Mag+"Looking for an ?x that matches:"+st)
    for f in facts:
        print("     ",f)
    bindings=kb.betterAsk(facts)
    if len(bindings) != 1:
        if object.text[-1] == "s":
            print(Mag+"Found reference in KB:")
            elements = [a for a in map(lambda x: x["?x"], bindings)]
            recentMemory.addElements(object.lemma_,elements)
            return elements
        else:
            print(Mag+"We have a problem with too many objects matching in the KB"+st)
    elif bindings == False:
        print(Mag+"We have a problem because nothing matches these facts in the KB"+st)
    else:
        print(Mag+"Found reference in KB:"+st,bindings[0]["?x"])
        recentMemory.addElements(object.lemma_,[bindings[0]["?x"]])
        return [bindings[0]["?x"]]

# discoverQuantifiedObjects builds out a FOPC desription of an object with its "name" as a variable
# and then uses "betterAsk" to find a binding that matches those facts. It uses the features in
# the text to build these statements.

def discoverQuantifiedObjects(object):

    count_number = utilities.countNumber(False,object)
    reference = languageTools.trackReference(object)

    print(Cyn+"Building concept to search KB for"+st,object.lemma_, "'"+reference.lemma_+"'")

    features = languageTools.extractFeatures(object)
    facts = [["inst","?x",reference.lemma_]]
    for f in features:
        type = kb.whatAmI(f.lemma_)
        facts.append([type,"?x",f.lemma_])
    print(rd+"Looking for an ?x that matches:"+st)
    for f in facts:
        print("     ",f)
    bindings=kb.betterAsk(facts)
    if len(bindings) < count_number:
        print(Cyn+"We have a problem with too few objects matching in the KB"+st)
    elif bindings == False:
        print(Cyn+"We have a problem because nothing matches these facts in the KB"+st)
    else:
        print(Cyn+"Found references in KB:")
        print(Cyn+"Selecting a subset from",len(bindings),"elements")
        candidates = [a for a in map(lambda x: x["?x"], bindings)]
        print(longTermMemory)
        remaining = longTermMemory.getElementAndStatus(reference.lemma_, "remaining")
        if remaining:
            candidates = [a for a in filter(lambda x: x in remaining, candidates)]
            print(Cyn+"Filtering off of last statement")
        names = candidates[0:count_number]
        others = candidates[count_number:]
        print(Cyn+"Selected"+st, names)
        recentMemory.addElements(reference.lemma_, names)
        recentMemory.addElementAndStatus(reference.lemma_, others, "remaining")
        return names

# discoverReferencedObjects builds out a FOPC desription of an object with its "name" as a variable
# and then uses "betterAsk" to find a binding that matches those facts. It uses the features in
# the text to build these statements.

def discoverReferencedObjects(object):
    if object.lemma_ in ["other", "rest"]: return longTermMemory.remaining()


# buildAndAssertOBject just checks to see if the thing mentioned is a class in the KB and adds it to the
# KB as an element of the class
def buildAndAssertObjectStatementFOPC(object, names):

    if kb.Category(object.lemma_):
        for name in names:
            kb.assertFromStatement(["inst", name, object.lemma_])
            statements.append(["inst", name, object.lemma_])


def findAndAssertFeaturesFOPC(object,names):
    print(Ylw+"Checking features to modify"+st, names)
    features = languageTools.extractFeatures(object)
    for feature in features:
        print(Ylw+"Found"+st,feature)
        feature_type = kb.whatAmI(feature.lemma_)
        for name in names:
            kb.assertFromStatement([feature_type, name, feature.lemma_])
            statements.append([feature_type, name, feature.lemma_])


def findAndAssertDefinitionsFOPC(object,names):
    print(Ylw+"Checking features to define"+st, names)
    direct = languageTools.extractDirectObject(object)
    if direct:
        print(Ylw+"Found defining feature"+st,direct)
        if kb.Category(direct.lemma_):
            for name in names:
                kb.assertFromStatement(["inst", name, direct.lemma_])
                statements.append(["inst", name, direct.lemma_])


# Starting with an object, pull off its prepositions and build FOPC statements that link their objects
# to the original object. We hand in the source of the prepositions (it could be either the verb or the
# noun that it dominates) and the object that it will modify.
def findAndAttachPrepObjectsFOPC(source,object,names):
    print(Ylw+"Checking pObjs of"+st,source.lemma_,Ylw+"to modify"+st, object.lemma_)

    preps = languageTools.extractPreps(source)
    prep_pairs = [a for a in map(lambda x: [x, languageTools.extractPrepObject(x)], preps)]

    for f in prep_pairs:
        print(Ylw+"  vPrep:"+st, f[0], Ylw+"Feature:"+st, f[1])
        prepObject = resolveObjectFOPC(f[1])[0]
        print(f[1],"resolved to",prepObject)
        for name in names:
            statements.append([f[0].lemma_,name,prepObject])
            kb.assertFromStatement([f[0].lemma_,name,prepObject])

    print(Ylw+"Done with prepositions for:"+st,object.lemma_)
