- info
    - lv1
    - 해시

# 완주하지 못한 선수
## 문제 설명
마라톤 대회에 참가한 선수들의 이름이 담긴 배열 `participant`와 완주한 선수들의 이름이 담긴 배열 `completion`이 주어집니다.

완주하지 못한 단 한 명의 선수 이름을 return 하도록 solution 함수를 완성해주세요.

---

## 제한 사항

- 참가자 수는 1명 이상 100명 이하입니다.
- `completion`의 길이는 `participant`의 길이보다 1 작습니다.
- 참가자의 이름은 알파벳 소문자로만 이루어져 있습니다.
- 참가자 중에는 동명이인이 있을 수 있습니다.

---

## 입출력 예

| participant                             | completion                       | result  |
| --------------------------------------- | -------------------------------- | ------- |
| ["leo", "kiki", "eden"]                 | ["eden", "kiki"]                 | "leo"   |
| ["marina", "josipa", "nikola", "vinko"] | ["josipa", "nikola", "marina"]   | "vinko" |
| ["mislav", "stanko", "mislav", "ana"]   | ["stanko", "ana", "mislav"]      | "mislav"|

---

## 입출력 설명
- 첫 번째 예시에서 "leo"는 참가했지만 완주하지 못했습니다.
- 두 번째 예시에서 "vinko"는 참가했지만 완주하지 못했습니다.
- 세 번째 예시에서 "mislav"는 두 명이 참가했지만 한 명만 완주했습니다.
