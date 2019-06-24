# Assignment 5: Natural Language Processing - Adding Features


## Introduction

In this assignment, you will implement the construction of new features 
based on statements such as:

> The blue block is a cube. \
> The block on the table is red.

You will need both spaCy and a knowledge base. 

## Your Task

You need to complete the code stub given to you in `student_code.py`. 
Test code is again in `main.py`. Notice that a global KB 
is maintained in `kb.py`, and its state of the KB persists between tests. 
This means that if your code fails one of the tests, say a test that 
involves generating facts of 10 blocks, then future tests involving blocks may 
also fail, since the serial number of the block may be thrown off.     

Note that the KB used in this assignment is different from the ones you 
have been using. 

## The Code Base

The knowledge base is defined in two files:

 
`kb.py` – The code for a basic knowledge base and inference engine. Along with Assert, and Ask, you will have betterAsk(a better version of Ask), whatAmI(takes a term and returns its TYPE from the KB, and howMany(takes a list of statements and returns the number of elements for which that statement is true). \
`reader.py` – The code for parsing the file of statements and rules. 


A core list of statements is in:

`initial_kb.txt` – A file with statements and rules.


You will also have a file with a set of functions around spaCy.  spaCy will process a statement and provide a parse tree.  The secondary functions will give you specific features from the tree. These are all in `languageTools.py`.

The functions include:

**extractRoot(document)**: Given a document provided by spaCy, extract its root. \
**extractExistentialTarget(token)**: Given a token associated with the verb “to be”, extract the existential target that.  In “There is a tree”, “tree” is what the statement is about \
**extractTarget(token)**: For statements about features, extract the things that is getting the feature. In “The tree is tall”, “tree” is the target. \
**extractDirectObject(token)**: Get the direct object associated with a verb. \
**extractPreps(token)**: Get the prepositions (the HEAD of phrases) associated with a verb or noun \
**extractPrepsAll(token)**: Get the prepositions (the HEAD of phrases) associated with a verb and its direct object \
**extractObjectFromPrep(token)**: Get the object of a preposition \
**extractPrepObjects(token)**: Get the objects associated with all of the prepositions modifying a verb or noun \
**extractPrepObjectsAll(token)**: Get the objects associated with all of the prepositions modifying a verb and its direct object \
**extractFeatures(token)**: Extract all of the features modifying an object \
**extractDeterminer(token)**: Extract all of the determiner of an object \
**extractCount(token)**: Extract any number modifying an object \
**extractQuestionDetail(token)**: Extract the type of a question from the ROOT. \
**determineObjectType(token)**: Just hands back the part of speech (normalized into English) of a token. 

You also have two display functions:

**showSet(sentences)**: Takes a list of sentences, hands them to spaCy and then pretty prints the result. \
**showTree(document)**: Takes a spaCy document and pretty prints it \
**showToken(token)**: Given a spaCy token (word), pretty print it 


There is also a file, `class.py`, that sets up the knowledge base and runs a set of sentences.


