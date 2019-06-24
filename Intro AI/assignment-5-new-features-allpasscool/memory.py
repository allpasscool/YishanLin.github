class memory:
    
    def __init__(self):
        self.mymemory = {}
    
    def addElementAndStatus(self, key, elements, status):
        if key in self.mymemory.keys():
            self.mymemory[key][status] = elements
        else:
            self.mymemory[key] = {}
            self.mymemory[key][status] = elements

    def addElements(self, key, elements):
        self.addElementAndStatus(key, elements, "referenced")

    def getElements(self, key):
        if key in self.mymemory.keys():
            if "referenced" in self.mymemory[key].keys():
                return self.mymemory[key]["referenced"]

    def remaining(self):
        for key in self.mymemory.keys():
            if "remaining" in self.mymemory[key].keys():
                return self.mymemory[key]["remaining"]
    
    def getElementAndStatus(self, key, status):
        if key in self.mymemory.keys():
            basic = self.mymemory[key]
            print(basic)
            if status in basic.keys():
                return basic[status]
    
    def __str__(self):
        return "Category: " + str(self.mymemory)
