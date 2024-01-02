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


# 5
def solution(data):
    return "".join(map(str, data)).count("1")


# 6
def solution(data):
    return sum(map(int, filter(str.isdigit, data)))


# 7
def solution(data):
    return list(map(lambda x: x[0], sorted(data, key=lambda x: x[1])))


# 8
def solution(data):
    return list(sorted(zip(data, data[1:]), key=lambda x: x[1] - x[0])[0])


# 9
def solution(data):
    return sorted(data, key=lambda x: x["수"], reverse=True)[0]["이름"]


# 10
def solution(data):
    return sorted(map(lambda x: x[0], filter(lambda x: sum(x[1:]) > 350, data)))


# 11
def solution(data):
    return len(list(filter(lambda x: sum(x) > 240, data)))


# 12
def solution(data):
    return sorted(data[0], key=lambda x: data[1].get(x.split(" ")[1]))


# 13
def solution(data):
    books, publish_years = data
    return sorted(books, key=lambda book: (publish_years[book], book))

