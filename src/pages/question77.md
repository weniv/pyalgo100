- info
    - lv0
    - 좌표평면

# 선 위에 점 확인
## 문제 설명
1차원 좌표상의 선분과 점이 주어집니다. 선분은 시작점과 끝점으로, 점은 하나의 좌표로 표현됩니다. 주어진 점이 선분 위에 있는지 여부를 확인하는 코드를 작성해주세요. 점이 선분 위에 있다면 True를, 그렇지 않다면 False를 반환합니다.

---

## 제한 사항

- 선분의 시작점과 끝점은 정수로 주어집니다.
- 점의 좌표 역시 정수로 주어집니다.
- 좌표의 범위는 0 이상 10,000 이하입니다.

---

## 입출력 예

| 입력 (선분, 점) | 출력 (결과) |
| --------------- | ---------- |
| ([1, 5], 3) | True |
| ([2, 8], 1) | False |
| ([10, 20], 15) | True |

---

## 입출력 설명
주어진 점이 선분 위에 있는지 여부를 확인합니다. 예를 들어, 선분 [1, 5]에 점 3이 있다면 True를, 선분 [2, 8]에 점 1이 있다면 False를 반환합니다.