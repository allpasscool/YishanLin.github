
syms = dict()


def genName(name = "Sym"):
    name = name.capitalize()
    if name in syms.keys():
        syms[name] = syms[name]+1
    else:
        syms[name] = 0
    return name+str(syms[name])

def number2number(string):
    try:
        num = int(string)
        return num
    except:
        if string in numbers.keys():
            return numbers[string]
        else:
            return 10

def genNames(name,number):
    return[a for a in map(lambda n: genName(name), range(number))]
	
def countNumber(det,n):
    numbers = {"one" : 1, "two" : 2, "three" : 3, "four" : 4, "five" : 5}
    if n: n=n.lemma_
    if det: return 1
    elif n in numbers.keys(): return numbers[n]
    else: return 10

