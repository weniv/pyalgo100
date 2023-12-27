# 1
def solution(data):
    return sum(filter(lambda x: x % 2, data))


# 2
def solution(data):
    if not data:
        return 0
    result = 1
    for x in data:
        if x == 0:
            return 0
        result *= x
    return result


# 3
def solution(data):
    return sum(filter(lambda x: not (x % 3 == 0 or x % 5 == 0), data))


# 4
def solution(data):
    result = 0
    for i, s in enumerate(data):
        number = int(s.split(" ")[1].replace("개", ""))
        result += number * (i + 1)
    return result
