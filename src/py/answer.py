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
        hh, mm, ampm = time.split(" ")[0].split(":") + time.split(" ")[1:]

        # 12시간제를 24시간제로 변환합니다. 12:00 AM은 00:00으로, 12:00 PM은 12:00으로 처리합니다.
        if ampm == "PM" and hh != "12":
            hh = str(int(hh) + 12)
        elif ampm == "AM" and hh == "12":
            hh = "00"

        return hh + ":" + mm + " " + ampm

    # 변환된 시간을 오름차순으로 정렬합니다.
    return sorted(times, key=convert_time)


# 16
def solution(dates):
    def convert_date(date):
        # 날짜 구분자에 따라 날짜를 분리합니다.
        if "-" in date:
            day, month, year = date.split("-")
        elif "/" in date:
            month, day, year = date.split("/")
        else:  # '.' 구분자
            year, month, day = date.split(".")

        return year, month, day

    # 날짜들을 연/월/일 형식으로 변환합니다.
    converted_dates = [convert_date(date) for date in dates]

    # 변환된 날짜들을 오름차순으로 정렬합니다.
    sorted_dates = sorted(converted_dates)

    # 정렬된 날짜들을 '연/월/일' 형식으로 다시 변환합니다.
    return ["/".join(date) for date in sorted_dates]


# 17
# 모듈 사용
from datetime import datetime


def solution(schedules):
    # 모든 일정을 하나의 리스트로 변환하면서 요일 정보를 함께 저장합니다.
    all_schedules = []
    for day, dates in schedules.items():
        for date in dates:
            all_schedules.append((date, day))

    # 날짜를 기준으로 내림차순 정렬합니다.
    all_schedules.sort(key=lambda x: x[0], reverse=True)

    # 최근 3개의 일정을 선택합니다.
    recent_three = all_schedules[:3]

    # 선택된 일정을 'YY-MM-DD 요일' 형식으로 변환합니다.
    return [
        datetime.strptime(date, "%Y-%m-%d").strftime("%y-%m-%d") + " " + day
        for date, day in recent_three
    ]


# test
from datetime import datetime

datetime.strptime("2024-01-01", "%Y-%m-%d").strftime("%y-%m-%d")


# 모듈 사용하지 않음
def solution(schedules):
    # 모든 일정을 하나의 리스트로 변환하면서 요일 정보를 함께 저장합니다.
    all_schedules = []
    for day, dates in schedules.items():
        for date in dates:
            # 날짜를 'YYYY-MM-DD'에서 'YY-MM-DD 요일' 형식으로 변환합니다.
            converted_date = date[2:] + " " + day
            all_schedules.append(converted_date)

    # 변환된 일정을 내림차순으로 정렬합니다.
    all_schedules.sort(reverse=True)

    # 최근 3개의 일정을 선택합니다.
    return all_schedules[:3]


def solution(data):
    all_schedules = []

    for day, dates in data.items():
        for date in dates:
            converted_date = f"{date[2:]} {day}"
            all_schedules.append(converted_date)
    all_schedules.sort(reverse=True)
    return all_schedules[:3]


# 18
def solution(temperature_data):
    # 온도 데이터를 (온도, 날짜) 형식의 튜플 리스트로 변환합니다.
    temp_list = [(temp, date) for date, temp in temperature_data.items()]

    # 온도를 기준으로 내림차순 정렬하되, 온도가 같은 경우 날짜를 기준으로 오름차순 정렬합니다.
    temp_list.sort(key=lambda x: (-x[0], x[1]))

    # 최고 온도 상위 3일을 선택합니다.
    top_three = temp_list[:3]

    # 선택된 날짜를 'YY-MM-DD: 온도' 형식으로 변환합니다.
    return [date[2:] + ": " + str(temp) for temp, date in top_three]


# 테스트
test_data = {
    "2024-01-01": 15,
    "2024-01-02": 17,
    "2024-01-03": 16,
    "2024-01-04": 20,
    "2024-01-05": 19,
    "2024-01-06": 21,
    "2024-01-07": 18,
}
print(solution(test_data))


def solution(data):
    temp_list = [(temp, date) for date, temp in data.items()]
    temp_list.sort(key=lambda x: (-x[0], x[1]))
    top_three = temp_list[:3]
    return [f"{date[2:]}: {temp}" for temp, date in top_three]


# 19
def solution(data):
    return [type(i).__name__ for i in data]


# 20
def solution(data):
    return all([type(instance).__name__ == class_ for class_, instance in data])


# 21
def solution(data):
    arr, target = data
    return arr.index(target) if target in arr else False


# 22
def solution(data):
    s, target = data
    return s.index(target) if target in s else False


# 23
def solution(data):
    matrix, target = data
    # 각 행을 순회하며 타겟 숫자 탐색
    for row in matrix:
        if target in row:
            return True
    return False


def solution(data):
    matrix, target = data
    return any([target in row for row in matrix])


def solution(data):
    matrix, target = data
    return target in sum(matrix, [])


# 24
# 카데인 알고리즘
# 현재까지의 최대 합과 현재 위치에서의 최대합을 갱신합니다.
# 이 알고리즘은 현재 위치까지의 최대합이 음수인 경우 현재 위치부터 다시 더하기 시작하는 알고리즘입니다.
# 음수가 아니면 더해나가는 것이 최대합이 되는 것을 응용한 것입니다.
def solution(data):
    if not data:
        return 0

    max_sum = current_sum = data[0]

    for num in data[1:]:
        print(num)
        # 현재 값인 num과 현재 위치까지의 최대합과 num을 더한 값 중 큰 값을 현재 위치까지의 최대합으로 설정합니다. 이전까지 더한 값이 음수인 경우 '현재 값'부터 다시 더하기 시작합니다.
        current_sum = max(num, current_sum + num)
        # 이렇게 설정된 현재 위치까지의 최대합을 최대 합과 비교하여 더 큰 값을 최대 합으로 설정합니다.
        max_sum = max(max_sum, current_sum)
        print("-----------")

    return max_sum


# 25
def solution(N):
    prime = [True] * (N + 1)
    prime[0] = prime[1] = False
    p = 2

    while p * p <= N:
        if prime[p]:
            for i in range(p * p, N + 1, p):
                prime[i] = False
        p += 1

    return sum(prime)


# 26
def solution(data):
    nums, k = data
    # 윈도우의 초기 합계 계산
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # 슬라이딩 윈도우를 이용해 최대 합계 탐색
    for i in range(len(nums) - k):
        window_sum = window_sum - nums[i] + nums[i + k]
        max_sum = max(max_sum, window_sum)

    return max_sum


# 27
def solution(data):
    nums, s = data
    min_length = float("inf")
    window_sum = 0
    window_start = 0

    for window_end in range(len(nums)):
        window_sum += nums[window_end]

        # 윈도우의 합이 s 이상이 되면 시작점을 이동시키면서 최소 길이를 갱신
        while window_sum >= s:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= nums[window_start]
            window_start += 1

    return min_length if min_length != float("inf") else 0


# 28
def solution(data):
    nums, target = data
    closest_sum = float("inf")
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]
        # 타겟 값에 더 가까운 합을 찾으면 업데이트
        if abs(target - current_sum) < abs(target - closest_sum):
            closest_sum = current_sum

        # 포인터 이동
        if current_sum < target:
            left += 1
        else:
            right -= 1

    return closest_sum


# 29
def solution(nums):
    result = 0
    for num in nums:
        result ^= num
    return result


# 30
def solution(data):
    return bin(data)[2:].replace("0", "B").replace("1", "A")


# 31
def solution(n):
    # 10자리 이진수로 변환
    binary_str = format(n, "010b")
    """
    b: 이진수(binary) 형식을 나타냅니다. 이 옵션은 정수를 이진수 형태의 문자열로 변환합니다.

    010: 이 형식 지정자는 출력될 이진수의 총 길이를 10자리로 지정합니다. 이는 출력되는 이진수 문자열이 10자리가 되도록 합니다. 만약 이진수의 길이가 10자리보다 짧다면, 앞쪽에 '0'을 추가하여 길이를 10자리로 만듭니다.
    """
    print(binary_str)
    # 비트 반전
    inverted_binary_str = "".join("1" if bit == "0" else "0" for bit in binary_str)
    # 반전된 이진수를 다시 정수로 변환
    return int(inverted_binary_str, 2)


def solution(n):
    # 이진수로 변환하고 '0b' 제거 후 zfill 메서드를 이용하여 10자리로 만들기
    binary_str = bin(n)[2:].zfill(10)
    # 비트 반전
    inverted_binary_str = "".join("1" if bit == "0" else "0" for bit in binary_str)
    # 반전된 이진수를 다시 정수로 변환
    return int(inverted_binary_str, 2)


# 예시 실행
print(solution(5))  # 예상 결과: 1018
print(solution(9))  # 예상 결과: 1014
print(solution(0))  # 예상 결과: 1023
print(solution(15))  # 예상 결과: 1008
print(solution(1023))  # 예상 결과: 0


# 32
def solution(nums):
    if not nums:
        return (0, 0)

    bit_and = nums[0]
    bit_or = nums[0]

    for num in nums[1:]:
        bit_and &= num
        bit_or |= num

    return (bit_and, bit_or)


# 33
import re


def solution(data):
    pattern = r"^[a-zA-Z0-9._+]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, data))


# 34
import re


def solution(data):
    pattern = r"(\d{4})-(\d{2})-(\d{2})"
    matches = re.findall(pattern, data)
    return [(int(year), int(month), int(day)) for year, month, day in matches]


# 35
import re


def solution(data):
    pattern = r"<[^>]+>"
    return re.sub(pattern, "", data)


# 36
import re


def solution(data):
    pattern = r"(\d{3})(\d{3,4})(\d{4})"
    return re.sub(pattern, r"\1-\2-\3", data)


# 37
import re


def solution(data):
    pattern = r"\[(?P<time>\d{2}:\d{2}:\d{2})\] (?P<message>.+)"
    match = re.match(pattern, data)
    if match:
        return match.groupdict()
    else:
        return None


# 37
import re


def solution(data):
    pattern = r"\[(?P<time>\d{2}:\d{2}:\d{2})\] (?P<message>.+)"
    match = re.match(pattern, data)
    if match:
        time = match.group("time")
        message = match.group("message")
        return {"time": time, "message": message}
    else:
        return None


# 38
import re


# 38
import re


def solution(data):
    pattern = r"(?P<protocol>https?|ftp)://(?P<domain>[^/\s]+)(?P<path>/[^?]*|)(\?(?P<query>[^#\s]*))?"
    match = re.match(pattern, data)
    if match:
        return {
            "protocol": match.group("protocol"),
            "domain": match.group("domain"),
            "path": match.group("path") if match.group("path") else "",
            "query": match.group("query") if match.group("query") else "",
        }
    else:
        return None


# 39
import re


def solution(data):
    pattern = r".*\.([^./]+)$"
    match = re.search(pattern, data)
    if match:
        return match.group(1)
    else:
        return ""


# 39
def solution(data):
    parts = data.split(".")
    if len(parts) > 1 and parts[-1] != "":
        return parts[-1]
    else:
        return ""


# 40
import re


def solution(data):
    numbers_and_commas = re.findall(r"[\d,]+", data)
    extracted_numbers = "".join(numbers_and_commas).replace(",", "")
    return extracted_numbers


# 40
def solution(data):
    return "".join([char for char in data if char.isdigit()])


# 41
def solution(data):
    stack = []
    bracket_map = {"(": ")", "{": "}", "[": "]"}

    for char in data:
        if char in bracket_map:
            stack.append(char)
        elif stack and char == bracket_map[stack[-1]]:
            stack.pop()
        else:
            return False

    return not stack


# 42
from collections import deque


def solution(data):
    size = data["size"]
    requests = data["requests"]
    queue = deque(maxlen=size)
    for request in requests:
        queue.append(request)
    return list(queue)


# 43
from collections import OrderedDict


def solution(data):
    size = data["size"]
    pages = data["pages"]
    cache = OrderedDict()

    for page in pages:
        if page in cache:
            cache.pop(page)
        elif len(cache) >= size:
            cache.popitem(last=False)
        cache[page] = True
    return list(cache.keys())


# 44
def solution(data):
    words = data.lower().split()
    frequency = {}
    for word in words:
        cleaned_word = "".join(char for char in word if char.isalpha())
        if cleaned_word:
            frequency[cleaned_word] = frequency.get(cleaned_word, 0) + 1
    return frequency


# 45
from collections import deque


def solution(data):
    queue1 = deque(data["queue1"])
    queue2 = deque(data["queue2"])
    sum1, sum2 = sum(queue1), sum(queue2)
    total_sum = sum1 + sum2
    operations = 0

    if total_sum % 2 != 0:
        return -1

    while sum1 != sum2:
        if sum1 > sum2:
            value = queue1.popleft()
            sum1 -= value
            sum2 += value
            queue2.append(value)
        else:
            value = queue2.popleft()
            sum2 -= value
            sum1 += value
            queue1.append(value)
        operations += 1

        if operations > len(queue1) + len(queue2):
            return -1

    return operations


# 46
from collections import deque


def solution(data):
    size = data["size"]
    commands = data["commands"]
    queue = deque(maxlen=size)
    result = []

    for command in commands:
        if command.startswith("insert"):
            _, element = command.split()
            if len(queue) == queue.maxlen:
                queue.popleft()
            queue.append(element)
            result.append(None)
        elif command == "delete":
            if queue:
                queue.popleft()
            result.append(None)
        elif command.startswith("search"):
            _, element = command.split()
            result.append(element in queue)

    return result


# 46 클래스 구현 문제
def solution(data):
    class CircularQueue:
        def __init__(self, size):
            self.queue = []
            self.size = size

        def insert(self, element):
            self.queue.append(element)
            if len(self.queue) > self.size:
                self.queue.pop(0)

        def delete(self):
            if self.queue:
                self.queue.pop(0)

        def search(self, element):
            return element in self.queue

    cq = CircularQueue(data["size"])
    result = []
    for command in data["commands"]:
        if command.startswith("insert"):
            _, element = command.split()
            cq.insert(element)
            result.append(None)
        elif command == "delete":
            cq.delete()
            result.append(None)
        elif command.startswith("search"):
            _, element = command.split()
            result.append(cq.search(element))

    return result


# 47


def solution(data):
    def find_max_depth(tree, index=0):
        if index >= len(tree) or tree[index] is None:
            return 0
        left_depth = find_max_depth(tree, 2 * index + 1)
        right_depth = find_max_depth(tree, 2 * index + 2)
        return max(left_depth, right_depth) + 1

    return find_max_depth(data)


# 47
def solution(tree):
    length = len(tree)
    depth = 0
    count = 1
    while True:
        if count > length:
            break
        count *= 2
        depth += 1
    return depth


# 48
def solution(data):
    tree = data["tree"]
    if not tree:
        return []

    stack = [(tree, tree["value"])]
    path_sums = []

    while stack:
        current, current_sum = stack.pop()

        # 현재 노드가 리프 노드인 경우, 경로 합을 결과에 추가
        if not current.get("left") and not current.get("right"):
            path_sums.append(current_sum)

        # 오른쪽 자식이 있으면 스택에 추가
        if current.get("right"):
            stack.append((current["right"], current_sum + current["right"]["value"]))

        # 왼쪽 자식이 있으면 스택에 추가
        if current.get("left"):
            stack.append((current["left"], current_sum + current["left"]["value"]))

    return path_sums


# 49
from collections import deque


def solution(data):
    def bfs_shortest_path(graph, start, end):
        visited = set()
        queue = deque([(start, 0)])  # (current node, distance)

        while queue:
            current, distance = queue.popleft()
            if current == end:
                return distance

            if current not in visited:
                visited.add(current)
                for neighbor in graph.get(current, []):
                    queue.append((neighbor, distance + 1))

        return -1  # Path not found

    graph = data["graph"]
    start = data["start"]
    end = data["end"]
    return bfs_shortest_path(graph, start, end)


# 50
def solution(data):
    def has_cycle(graph):
        visited = set()
        rec_stack = set()

        for node in graph:
            if node not in visited:
                if dfs(graph, node, visited, rec_stack):
                    return True

        return False

    def dfs(graph, current, visited, rec_stack):
        if current not in visited:
            visited.add(current)
            rec_stack.add(current)

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    if dfs(graph, neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

        rec_stack.remove(current)
        return False

    return has_cycle(data["graph"])


# 51
def solution(data):
    def min_coins(coins, amount):
        coins.sort(reverse=True)
        count = 0
        for coin in coins:
            count += amount // coin
            amount %= coin
            if amount == 0:
                break
        return count

    return min_coins(data["coins"], data["amount"])


# 52
def solution(data):
    total_cost = data * 700
    change = 10000 - total_cost

    denominations = [10000, 5000, 1000, 500, 100]
    change_list = [0, 0, 0, 0, 0]

    for i, denom in enumerate(denominations):
        change_list[i], change = divmod(change, denom)

    return change_list


# 53
def solution(data):
    investments, capital = data
    investments.sort(key=lambda x: x[0])  # 투자 금액 기준으로 정렬

    selected_companies = []
    for cost, company in investments:
        if capital >= cost:
            selected_companies.append(company)
            capital -= cost

    return selected_companies


# 54
def solution(data):
    room, path = data
    rows, cols = len(room), len(room[0])
    directions = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cleaned_count = 0
    x, y = 0, 0

    if room[0][0] == 1:
        cleaned_count += 1
        room[0][0] = 0

    for step in path:
        dx, dy = directions[step]
        nx, ny = x + dx, y + dy

        # 경로가 방 안에 있고, 아직 청소되지 않은 칸이라면 청소
        if 0 <= nx < rows and 0 <= ny < cols and room[nx][ny] == 1:
            cleaned_count += 1
            room[nx][ny] = 0  # 청소된 상태로 표시

        # 로봇 위치 업데이트
        if 0 <= nx < rows and 0 <= ny < cols:
            x, y = nx, ny

    return cleaned_count


# 55
def solution(data):
    matrix = data
    mine_count = 0

    for row in matrix:
        mine_count += row.count(1)

    return mine_count


# 56
def solution(matrix):
    mine_locations = []
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] == 1:
                mine_locations.append((row, col))
    return mine_locations


# 56
import numpy as np


def solution(matrix):
    matrix_np = np.array(matrix)
    return list(zip(*np.where(matrix_np == 1)))


# 57
def solution(data):
    matrix, div = data
    condition = lambda x: (x % div) == 0
    for row in matrix:
        if not all(condition(value) for value in row):
            return False
    return True


# 58
def solution(matrix):
    if not matrix or len(matrix) != len(matrix[0]):
        return "Error"
    return [list(reversed(col)) for col in zip(*matrix)]


# 59
def solution(data):
    matrix, condition = data
    count = 0
    total_sum = 0
    for row in matrix:
        for item in row:
            if item >= condition:
                count += 1
                total_sum += item
    return count, total_sum


# 테스트 케이스 적용 예시
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
condition1 = 5
result1 = solution(matrix1, condition1)
print(result1)  # (4, 30)


# 59
import numpy as np


def solution(matrix, condition):
    np_matrix = np.array(matrix)
    filtered_elements = np_matrix[np_matrix >= condition]
    count = filtered_elements.size
    total_sum = filtered_elements.sum()
    return count, total_sum


# 테스트 케이스 적용 예시
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
condition1 = 5
result1 = solution(matrix1, condition1)
print(result1)  # (4, 30)


# 60
def solution(data):
    matrix, range_values = data
    min_value, max_value = float("inf"), float("-inf")
    lower_bound, upper_bound = range_values

    for row in matrix:
        for item in row:
            if lower_bound <= item <= upper_bound:
                min_value = min(min_value, item)
                max_value = max(max_value, item)

    return (max_value, min_value) if min_value != float("inf") else (None, None)


# 테스트 케이스 적용 예시
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
range1 = (3, 7)
result1 = solution(matrix1, range1)
print(result1)  # (7, 3)


import numpy as np


# 60
def solution(data):
    matrix, range_values = data
    np_matrix = np.array(matrix)
    lower_bound, upper_bound = range_values

    # 조건에 맞는 요소들을 필터링
    filtered_elements = np_matrix[
        (np_matrix >= lower_bound) & (np_matrix <= upper_bound)
    ]

    if filtered_elements.size == 0:
        return (None, None)

    max_value = np.max(filtered_elements)
    min_value = np.min(filtered_elements)
    return (max_value, min_value)


# 테스트 케이스 적용 예시
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
range1 = (3, 7)
result1 = solution(matrix1, range1)
print(result1)  # (7, 3)


# 61
from collections import deque


def solution(data):
    deque_data, commands = data
    dq = deque(deque_data)
    for command in commands:
        direction, count = command
        if direction == "왼쪽":
            for _ in range(min(count, len(dq))):
                dq.popleft()
        elif direction == "오른쪽":
            for _ in range(min(count, len(dq))):
                dq.pop()
    return list(dq)


# 테스트 케이스 적용 예시
deque_data1 = [1, 2, 3, 4, 5]
commands1 = [("왼쪽", 2), ("오른쪽", 1)]
result1 = solution(deque_data1, commands1)
print(result1)  # [3, 4]


# 62
from collections import deque


def solution(data):
    max_size, nums = data
    dq = deque(maxlen=max_size)
    result = []

    for num in nums:
        dq.append(num)
        result.append(list(dq))

    return result


# 테스트 케이스 적용 예시
data1 = [3, [1, 2, 3, 4, 5]]
result1 = solution(data1)
print(result1)  # [[1], [1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
