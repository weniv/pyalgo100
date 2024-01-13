- info
    - lv1
    - 데이터 구조

# 순환 큐에서의 데이터 처리
## 문제 설명
주어진 크기의 순환 큐를 구현하고, 제공된 일련의 데이터 처리 명령(삽입, 삭제, 탐색)을 수행하는 함수를 작성해주세요. 순환 큐는 맨 뒤에 도달하면 다시 맨 앞으로 돌아가는 특징이 있습니다. 명령은 "insert", "delete", "search"로 구성되며, 각 명령은 다음과 같이 동작합니다:

- "insert [element]": 큐에 요소를 삽입합니다. 큐가 가득 차 있으면, 가장 오래된 요소를 삭제하고 새 요소를 삽입합니다.
- "delete": 큐에서 가장 오래된 요소를 삭제합니다.
- "search [element]": 큐 내에 해당 요소가 있는지 확인하고, 있으면 True, 없으면 False를 반환합니다.

예를 들어, 큐의 크기가 3이고, 순서대로 ["insert 1", "insert 2", "insert 3", "insert 4", "search 3", "delete", "search 3"] 명령이 주어진다면, ["insert 4" 명령 후에는 큐에 [2, 3, 4]가 저장되고, "search 3"은 True, "delete" 후에는 [3, 4]가 남으며, 마지막 "search 3"은 여전히 True를 반환합니다.

---

## 제한 사항

- 큐의 크기는 1 이상 100 이하의 정수로 주어집니다.
- 처리해야 할 명령은 최대 100개까지 주어질 수 있습니다.

---

## 입출력 예

|   크기  |               입력 (처리 명령)                 | 출력 (각 명령의 결과)     |
| ------ | -------------------------------------------- | ------------------------ |
| 3      | ["insert 1", "insert 2", "insert 3", "insert 4", "search 3", "delete", "search 3"] | [None, None, None, None, True, None, True] |
| 2      | ["insert A", "insert B", "insert C", "search B"] | [None, None, None, True] |
| 4      | ["insert X", "delete", "search X"]           | [None, None, False]      |

---

## 입출력 설명
각 명령을 순서대로 수행한 결과를 반환합니다. "insert"와 "delete" 명령은 결과가 없으므로 None을, "search" 명령은 결과를 True 또는 False로 반환합니다.
