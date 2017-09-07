'''
This is my solutions for this
http://comet.lehman.cuny.edu/schneider/Fall17/CMP464/LearnPython_NumPy.html
Note: I am using python 2.7
'''

# 1: Sum the squares of the first 20 odd numbers.
print sum([(x*2-1)**2 for x in range(1,21)])
# or like http://www.geeksforgeeks.org/sum-of-squares-of-even-and-odd-natural-numbers/
def oddSqrSum(n):
    return n*(2*n+1)*(2*n-1)/3
print oddSqrSum(20)

# 2: Make a dictionary with keys being names of people and values being their height.
persons = {'johnson': 35, 'jim':20, 'james':21, 'jay':14, 'jaimy': 34, 'jones':1}
# 2.1: Make a function that returns the name of the tallest person (your function does not know the length)
def maxHeigh(persons=persons):
    maxV = persons.itervalues().next() #get first value
    person = ''
    for p in persons.keys():
        maxV, person= (persons.get(p), p) if maxV <= persons.get(p)  else (maxV, person)
    return person
print maxHeigh()
# 2.2: Make a function that returns a list of the names in the dictionary sorted by height. (use a different sort than in tutorial)
def sortDict(persons=persons):
    snosrep = {persons.get(key): key for key in persons.keys()}
    return [snosrep[sortHeight] for sortHeight in sorted(snosrep.keys())]
print sortDict()

# 3: You are given two vectors in an arbitrary N dimensional space.....
# 3.1: The distance between two vectors is defined as....
# 3.2: Use arrays and dot product to find the distance without using loops.
