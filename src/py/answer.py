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

# 14
def solution(data):
    sorted_data = sorted(data.keys())
    return [data[i] for i in sorted_data]

# 15
def solution(times):
    def convert_time(time):
        hh, mm, ampm = time.split(' ')[0].split(':') + time.split(' ')[1:]

        # 12시간제를 24시간제로 변환합니다. 12:00 AM은 00:00으로, 12:00 PM은 12:00으로 처리합니다.
        if ampm == 'PM' and hh != '12':
            hh = str(int(hh) + 12)
        elif ampm == 'AM' and hh == '12':
            hh = '00'

        return hh + ':' + mm + ' ' + ampm

    # 변환된 시간을 오름차순으로 정렬합니다.
    return sorted(times, key=convert_time)

# 16


def solution(dates):
    def convert_date(date):
        # 날짜 구분자에 따라 날짜를 분리합니다.
        if '-' in date:
            day, month, year = date.split('-')
        elif '/' in date:
            month, day, year = date.split('/')
        else:  # '.' 구분자
            year, month, day = date.split('.')

        return year, month, day
    # 날짜들을 연/월/일 형식으로 변환합니다.
    converted_dates = [convert_date(date) for date in dates]

    # 변환된 날짜들을 오름차순으로 정렬합니다.
    sorted_dates = sorted(converted_dates)

    # 정렬된 날짜들을 '연/월/일' 형식으로 다시 변환합니다.
    return ["/".join(date) for date in sorted_dates]

# 테스트
test_dates = ['20-01-2024', '12/15/2023', '2022.05.30']
print(solution(test_dates))

