import core, reader, kb

# rd = "\033[31;1m"
# em = "\033[4m"
# st = "\033[0m"
# lf = "\n"

rd = ""
em = ""
st = ""
lf = "\n"

sExistentials = []
sFeatures = []
qExistentials = []
qFeatures = []
qCounts = []
Commands = []

sExistentials.append('There are two rooms on the first floor')
sExistentials.append('There is a wooden table in one of the rooms')
sExistentials.append('There is a metal table in the other')
sExistentials.append('There are ten red blocks on the wooden table')
sExistentials.append('There are two green blocks')
sExistentials.append('There is a box on the metal table')
sExistentials.append('There is a box on Table0')
sExistentials.append('There are three big red blocks on the wooden table on the first floor')

sFeatures.append('Three of the red blocks are cubes')
sFeatures.append('The green blocks are pyramids')
sFeatures.append('The boxes are big')
sFeatures.append('The pyramids are on the metal table')

Commands.append('Move Block2 from the table onto the green block')
Commands.append('Put the red block on the green block')

qFeatures.append('Is the red block big')
qFeatures.append('What color are the blocks on the table')
qFeatures.append('Where is the red block')
qFeatures.append('Where is the red block')

qExistentials.append('Are there three red blocks on the table')
qCounts.append('How many red blocks are on the table')

qCounts.append('How many red blocks are on the table')

# Let's read in an initialize our knowledge base

facts, rules = reader.read_tokenize("initial_kb.txt")

print("\nBuilding world: facts\n")

for fact in facts:
    kb.Assert(kb.statement(fact))

print("\nBuilding world: rules\n")

for new_rule in rules:
    kb.Assert_Rule(kb.rule(new_rule[0], new_rule[1]))

core.buildFOPC(sExistentials[0])
core.buildFOPC(sExistentials[1])
core.buildFOPC(sExistentials[2])
core.buildFOPC(sExistentials[3])
core.buildFOPC(sExistentials[4])
core.buildFOPC(sExistentials[5])
core.buildFOPC(sExistentials[6])

core.buildFOPC(sFeatures[0])
core.buildFOPC(sFeatures[1])
core.buildFOPC(sFeatures[2])
core.buildFOPC(sFeatures[3])

for s in core.statements:
    print(s)
