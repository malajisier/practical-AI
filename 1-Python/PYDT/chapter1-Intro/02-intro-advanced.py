import time

start = time.time()

for a in range(0, 1001):
    for b in range(0, 1001):
        c = 1000 - a - b
        if a**2+b**2 == c**2:
            print('a: %d, b: %d, c: %d' % (a, b, c))

end = time.time()
print('elapsed: %f' % (end - start))
print('completed')

'''output:

a: 0, b: 500, c: 500
a: 200, b: 375, c: 425
a: 375, b: 200, c: 425
a: 500, b: 0, c: 500
elapsed: 2.145771
completed

'''