import languageTools, utilities, reader, kb, memory, core

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

#### Your code starts here
# buildFeatureStatement takes a parse tree and builds out a set of FOPC statements that can be added to the KB
# that add features to EXISTING objects.  These objects can be referenced in one of three ways:
# They can be Proper Nouns: "Table0 is big"
# They can be descriptions: "The big table is red"
# They can refer to something that has been mentioned: "It is wooden"
def buildFeatureStatement(tree):

    print("Stubbed out version of buildFeatureStatement")
    # As with core.buildExistentials, we want to get out the ROOT and the primary NOUN. But here, the noun will
    # the subject. You call languageTools.extractRoot on the parse tree

 ##### Your code to extractRoot here

    root = languageTools.extractRoot(tree)
    # To get the subject of a verb, we use languageTools.extractSubject on the ROOT

##### Your code to extractSubject here

    subject = languageTools.extractSubject(root)
    # Once we have the primary noun, we then want to resolve it, that is, figure out what it refers
    # to using core.resolveObjectFOPC. Resolve object will give us a list of names that are referred to by the
    # words in the text.  They will always be names of existings objects. You need the names for the
    # other functions

##### Your code to resolveObjectFOPC here -- this will build some FOPC and Assert it

    names = core.resolveObjectFOPC(subject)
    # Then we need to figure out what is going to modify it. To do this, we go back to our verb and for
    # any modifiers associted with it. These will either be prepositional phrases or adjectives.

    # To get the prepositional phrases, we can use core.findAndAttachPrepObjectsFOPC that takes the ROOT and
    # the names and will build any FOPC associated with prepositional objects it finds and assert it.

##### Your code to resolveObjectFOPC here -- this takes the root, the primary (suibject) and your names

    core.findAndAttachPrepObjectsFOPC(root, subject, names)
    # Next we pull out any features associated with the object and attach them to the object. We can use
    # core.findAndAssertFeaturesFOPC to do this.  Like core.findAndAttachPrepObjectsFOPC, it takes the ROOT and
    # a list of names and builds the FOPC associated with any adjectives it finds

##### Your code to findAndAssertFeaturesFOPC here -- this takes the root and your names

    core.findAndAssertFeaturesFOPC(root, names)

##### Your code to findAndAssertDefinitionsFOPC -- this takes the root and your names

    core.findAndAssertDefinitionsFOPC(root, names)