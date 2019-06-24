import unittest
import reader, kb, core

class UnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sExistentials = []
        cls.sFeatures = []
        cls.qExistentials = []
        cls.qFeatures = []
        cls.qCounts = []
        cls.Commands = []

        cls.sExistentials.append('There are two rooms on the first floor')
        cls.sExistentials.append('There is a wooden table in one of the rooms')
        cls.sExistentials.append('There is a metal table in the other')
        cls.sExistentials.append('There are ten red blocks on the wooden table')
        cls.sExistentials.append('There are two green blocks')
        cls.sExistentials.append('There is a box on the metal table')
        cls.sExistentials.append('There is a box on Table0')
        cls.sExistentials.append('There are three big red blocks on the wooden table on the first floor')

        cls.sFeatures.append('Three of the red blocks are cubes')
        cls.sFeatures.append('The green blocks are pyramids')
        cls.sFeatures.append('The boxes are big')
        cls.sFeatures.append('The pyramids are on the metal table')

        cls.Commands.append('Move Block2 from the table onto the green block')
        cls.Commands.append('Put the red block on the green block')

        cls.qFeatures.append('Is the red block big')
        cls.qFeatures.append('What color are the blocks on the table')
        cls.qFeatures.append('Where is the red block')
        cls.qFeatures.append('Where is the red block')

        cls.qExistentials.append('Are there three red blocks on the table')
        cls.qCounts.append('How many red blocks are on the table')

        cls.qCounts.append('How many red blocks are on the table')

        facts, rules = reader.read_tokenize("initial_kb.txt")

        # Building world: facts

        for fact in facts:
            kb.Assert(kb.statement(fact))

        # Building world: rules

        for new_rule in rules:
            kb.Assert_Rule(kb.rule(new_rule[0], new_rule[1]))

        # Build the world

        core.buildFOPC('There are two rooms on the first floor')
        core.buildFOPC('There is a wooden table in one of the rooms')
        core.buildFOPC('There is a metal table in the other')
        core.buildFOPC('There are ten red blocks on the wooden table')
        core.buildFOPC('There are two green blocks')
        core.buildFOPC('There is a box on the metal table')
        core.buildFOPC('There is a box on Table0')

    def test01_01(self):
        core.buildFOPC('Three of the red blocks are cubes')
        self.assertTrue(kb.betterAsk([['inst', 'Block0', 'cube']]))

    def test01_02(self):
        core.buildFOPC('Three of the red blocks are cubes')
        self.assertTrue(kb.betterAsk([['inst', 'Block1', 'cube']]))

    def test01_03(self):
        core.buildFOPC('Three of the red blocks are cubes')
        self.assertTrue(kb.betterAsk([['inst', 'Block2', 'cube']]))

    def test02_01(self):
        core.buildFOPC('The green blocks are pyramids')
        self.assertTrue(kb.betterAsk([['inst', 'Block10', 'pyramid']]))

    def test02_02(self):
        core.buildFOPC('The green blocks are pyramids')
        self.assertTrue(kb.betterAsk([['inst', 'Block11', 'pyramid']]))

    def test03_01(self):
        core.buildFOPC('The boxes are big')
        self.assertTrue(kb.betterAsk([['size', 'Box0', 'big']]))

    def test03_02(self):
        core.buildFOPC('The boxes are big')
        self.assertTrue(kb.betterAsk([['size', 'Box1', 'big']]))

    def test04_01(self):
        core.buildFOPC('The pyramids are on the metal table')
        self.assertTrue(kb.betterAsk([['on', 'Block10', 'Table1']]))

    def test04_02(self):
        core.buildFOPC('The pyramids are on the metal table')
        self.assertTrue(kb.betterAsk([['on', 'Block11', 'Table1']]))


if __name__ == '__main__':
    unittest.main()
