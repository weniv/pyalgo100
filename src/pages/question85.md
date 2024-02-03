- info
    - lv1
    - 자료 구조

# 응급환자 발생
## 문제 설명

실리콘벨리로 출장을 가는 도중 응급환자가 다수 발생했습니다. 현재 위치는 바다 한 가운데이며, 가장 빠른 시간 내에 환자를 치료할 수 있는 공항으로 이동해야 합니다. 각 공항과의 거리와 그 공항에서 치료할 수 있는 병명이 포함된 JSON 형태의 데이터 `airportData`와 현재 환자 목록을 담은 `patientData`가 주어집니다. 가장 최적의 경로를 찾아 방문하는 공항의 이름을 출력하는 코드를 작성하세요.

---

## 제한 사항

- `airportData`는 공항 이름, 공항까지의 거리(단위: km), 치료 가능한 병명 리스트를 포함합니다.
- `patientData`는 발생한 환자의 병명 리스트를 포함합니다.
- 모든 입력 데이터는 유효하며, 각 환자는 치료할 수 있는 모든 공항에 경유해가며 치료를 받아야 합니다.
- 여러 공항이 조건을 만족하는 경우, 거리가 가장 가까운 공항을 선택합니다.
- 각 거리는 다른 공항으로 이동해도 해당 거리가 유지된다고 가정합니다. 예를 들어 A 공항에서 B 공항으로 이동했을 경우 distance를 다시 계산하지 않습니다.

---

## 입출력 예

### 입력
```python
[
    [
        # airportData
        {
            "name": "Airport A", 
            "distance": 500, 
            "treatableDiseases": ["Disease A", "Disease B"]
        },
        {
            "name": "Airport B", 
            "distance": 300, 
            "treatableDiseases": ["Disease C"]
        },
        {
            "name": "Airport C", 
            "distance": 400, 
            "treatableDiseases": ["Disease B", "Disease C"]
        }
    ], 
    ["Disease A", "Disease C"] # patientData
]
```

### 출력
```python
"Airport B", "Airport A"
```

### 입출력 설명:
- "Disease A"와 "Disease C"를 치료할 수 있는 공항은 "Airport A"와 "Airport B"입니다. "Airport A"는 거리가 500km로 "Airport B"(300km)보다 멀어 가장 최적의 경로는 "Airport B"를 방문하고 "Airport A"를 방문하는 것입니다.