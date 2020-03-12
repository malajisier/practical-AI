# 如果 a+b+c=1000，且 a^2+b^2=c^2（a,b,c 为自然数），如何求出所有a、b、c可能的组合?

import time

start = time.time()

for a in range(0, 1001):
    for b in range(0, 1001):
        for c in range(0, 1001):
            if a+b+c == 1000 and a**2+b**2 == c**2:
                print('a: %d, b: %d, c: %d' % (a, b, c))

end = time.time()

print('elapsed: %f' % (end - start))
print('completed')


''' output:
a: 0, b: 500, c: 500
a: 200, b: 375, c: 425
a: 375, b: 200, c: 425
a: 500, b: 0, c: 500
elapsed: 335.428164
completed
'''