- info
    - lv0
    - 정규표현식

# URL 및 쿼리스트링 파싱
## 문제 설명
주어진 URL에서 프로토콜, 도메인, 경로, 그리고 쿼리스트링을 추출하는 함수를 작성해주세요. URL은 `프로토콜://도메인/경로?쿼리스트링` 형식을 따릅니다. 쿼리스트링은 선택적이며, 없는 경우도 고려해야 합니다.

예를 들어, URL "https://www.weniv.co.kr/path/to/resource?user=abc&lang=en"에서 함수는 `{'protocol': 'https', 'domain': 'www.weniv.co.k', 'path': '/path/to/resource', 'query': 'user=abc&lang=en'}`를 반환해야 합니다.

- 특이사항: 서비스는 모듈을 쓰지 못하게 되어 있으나 처음에 solution 코드 없이 `import re`만 입력하여 한 번 실행하면 그 다음 코드부터 `import re`를 하지 않아도 사용 가능합니다. solution 함수 내에 re 모듈을 사용하셔도 애러가 나지 않습니다. 코드 내에는 solution 함수만 있어야 하므로 이 2개가 있지 않도록 주의해주세요. 어려우신 분은 `제주코딩베이스캠프` 유튜브 채널 33번 문제 영상을 참고해주세요.

---

## 제한 사항

- URL은 항상 올바른 형식을 따릅니다.
- 프로토콜, 도메인, 경로, 쿼리스트링은 모두 문자열입니다.

---

## 입출력 예

|   입력 (URL 문자열)                                  | 출력 (프로토콜, 도메인, 경로, 쿼리스트링 딕셔너리)  |
| --------------------------------------------------- | --------------------------------------------------- |
| "http://www.weniv.co.k"                             | {'protocol': 'http', 'domain': 'www.weniv.co.k', 'path': '', 'query': ''} |
| "https://www.weniv.co.kr/path/to/resource?user=abc&lang=en" | {'protocol': 'https', 'domain': 'www.weniv.co.kr', 'path': '/path/to/resource', 'query': 'user=abc&lang=en'} |
| "ftp://ftp.weniv.co.kr/folder?page=1"               | {'protocol': 'ftp', 'domain': 'ftp.weniv.co.kr', 'path': '/folder', 'query': 'page=1'} |

---

## 입출력 설명
각 URL에서 프로토콜, 도메인, 경로, 쿼리스트링을 추출하여 딕셔너리 형태로 반환합니다.
