from typing import List


def multiplicate(a: List[int]) -> List[int]:
    total = a[0]
    for i in range(1, len(a)):
        total *= a[i]

    b = []
    for i in range(len(a)):
        t = total // a[i]
        b.append(t)

    return b


c = [1, 2, 3, 4]
d = multiplicate(c)
print(c)
print(d)
