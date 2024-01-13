# PAGE_NAME으로 바로 호출 가능하도록 0번째 값을 비워줌
# black formatter 적용하면 안됨(마지막에 콤마 생기는 것때문에 error 발생)
# question_info로 info 정보 분할로 해결하였음, fomatter 사용 가능
testcase_and_result = [
    {"que_number": 0, "lv": "", "kinds": "", "testcase": "", "result": ""},
    {
        "que_number": 1,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [[1, 2, 3, 4, 5], [], [1, 3, 5, 7, 9], [2, 3, 4], [2, 4, 6, 8]],
        "result": [9, 0, 25, 3, 0],
    },
    {
        "que_number": 2,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [[1, 2, 3, 4, 5], [], [1, 3, 5, 7], [2, 3, 4], [2, 4, 6, 8]],
        "result": [120, 0, 105, 24, 384],
    },
    {
        "que_number": 3,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [[1, 2, 3, 4, 5], [], [1, 3, 5, 7], [2, 3, 4], [2, 4, 6, 8]],
        "result": [7, 0, 8, 6, 14],
    },
    {
        "que_number": 4,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [
            ["쿠키 1개", "쿠키 2개", "쿠키 3개"],
            ["쿠키 2개", "쿠키 2개", "쿠키 3개"],
            ["쿠키 3개", "쿠키 3개", "쿠키 3개"],
            ["쿠키 3개", "쿠키 2개", "쿠키 5개"],
            ["쿠키 1개", "쿠키 1개", "쿠키 1개"],
        ],
        "result": [14, 15, 18, 22, 6],
    },
    {
        "que_number": 5,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [
            [1, 2, 3, 4, 5],
            [1, 11, 111, 1111],
            [3, 4, 1, 4, 5],
            [],
            [1, 11, 1, 11],
        ],
        "result": [1, 10, 1, 0, 6],
    },
    {
        "que_number": 6,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": ["1hel2lo3", "1q2w3e4r", "", "hello", "1234"],
        "result": [6, 10, 0, 0, 10],
    },
    {
        "que_number": 7,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [["A", 1], ["B", 2], ["C", 3]],
            [["A", 3], ["B", 2], ["C", 1]],
            [["A", 1], ["B", 3], ["C", 2]],
            [["A", 1], ["B", 3], ["C", 2], ["D", 5], ["E", 4]],
        ],
        "result": [
            ["A", "B", "C"],
            ["C", "B", "A"],
            ["A", "C", "B"],
            ["A", "C", "B", "E", "D"],
        ],
    },
    {
        "que_number": 8,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [1, 3, 5, 7, 8, 10, 12],
            [10, 20, 30, 35, 45, 55],
            [4, 8, 12, 13, 20, 24],
            [19, 20, 30, 40, 50, 60, 70],
        ],
        "result": [[7, 8], [30, 35], [12, 13], [19, 20]],
    },
    {
        "que_number": 9,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                {"이름": "A", "국": 30, "영": 38, "수": 67},
                {"이름": "B", "국": 95, "영": 21, "수": 98},
                {"이름": "C", "국": 92, "영": 33, "수": 32},
            ],
            [
                {"이름": "A", "국": 67, "영": 67, "수": 81},
                {"이름": "B", "국": 82, "영": 32, "수": 98},
                {"이름": "C", "국": 95, "영": 11, "수": 99},
            ],
            [
                {"이름": "A", "국": 33, "영": 64, "수": 37},
                {"이름": "B", "국": 92, "영": 38, "수": 89},
                {"이름": "C", "국": 31, "영": 98, "수": 100},
            ],
        ],
        "result": ["B", "C", "C"],
    },
    {
        "que_number": 10,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ["Licat", 98, 92, 85, 97],
                ["Mura", 95, 32, 51, 30],
                ["Binky", 98, 98, 51, 32],
            ],
            [
                ["Gray", 98, 92, 85, 97],
                ["Gom", 98, 30, 21, 60],
                ["Allosa", 98, 90, 99, 98],
            ],
            [["A", 10, 15, 20, 25], ["B", 30, 35, 41, 10], ["C", 18, 30, 29, 18]],
        ],
        "result": [["Licat"], ["Allosa", "Gray"], []],
    },
    {
        "que_number": 11,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [[98, 92, 85], [95, 32, 51], [98, 98, 51]],
            [[92, 85, 97], [30, 21, 60], [90, 99, 98], [0, 0, 0], [81, 80, 88, 83]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        "result": [2, 3, 0],
    },
    {
        "que_number": 12,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ["제주시 A동 한라산길 61", "제주시 B동 백록담길 63", "제주시 C동 사라봉길 31"],
                {"A동": 63007, "B동": 63010, "C동": 63002},
            ],
            [
                ["제주시 E동 한라산길 11", "제주시 D동 한라산길 101", "제주시 F동 한라산길 21"],
                {"E동": 63107, "D동": 63310, "F동": 63032},
            ],
            [
                ["제주시 AE동 한라산길 61", "제주시 FE동 백록담길 63", "제주시 BE동 사라봉길 31"],
                {"AE동": 63111, "FE동": 63132, "BE동": 63337},
            ],
        ],
        "result": [
            ["제주시 C동 사라봉길 31", "제주시 A동 한라산길 61", "제주시 B동 백록담길 63"],
            ["제주시 F동 한라산길 21", "제주시 E동 한라산길 11", "제주시 D동 한라산길 101"],
            ["제주시 AE동 한라산길 61", "제주시 FE동 백록담길 63", "제주시 BE동 사라봉길 31"],
        ],
    },
    {
        "que_number": 13,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ["Great Expectations", "Brave New World", "The Catcher in the Rye"],
                {
                    "Great Expectations": 1861,
                    "Brave New World": 1932,
                    "The Catcher in the Rye": 1951,
                },
            ],
            [
                ["To Kill a Mockingbird", "1984", "Animal Farm"],
                {"To Kill a Mockingbird": 1960, "1984": 1949, "Animal Farm": 1945},
            ],
            [
                ["The Great Gatsby", "Moby Dick", "Pride and Prejudice"],
                {
                    "The Great Gatsby": 1925,
                    "Moby Dick": 1851,
                    "Pride and Prejudice": 1813,
                },
            ],
        ],
        "result": [
            ["Great Expectations", "Brave New World", "The Catcher in the Rye"],
            ["Animal Farm", "1984", "To Kill a Mockingbird"],
            ["Pride and Prejudice", "Moby Dick", "The Great Gatsby"],
        ],
    },
    {
        "que_number": 14,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            {"BX32": "1984", "AX21": "Moby Dick", "CX14": "To Kill a Mockingbird"},
            {
                "ZA10": "Pride and Prejudice",
                "YA24": "The Great Gatsby",
                "XA33": "The Catcher in the Rye",
            },
            {
                "LD44": "Brave New World",
                "KC55": "Great Expectations",
                "MD66": "Animal Farm",
            },
        ],
        "result": [
            ["Moby Dick", "1984", "To Kill a Mockingbird"],
            ["The Catcher in the Rye", "The Great Gatsby", "Pride and Prejudice"],
            ["Great Expectations", "Brave New World", "Animal Farm"],
        ],
    },
    {
        "que_number": 15,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            ["01:00 PM", "11:30 AM", "12:45 PM", "09:00 AM", "12:00 AM"],
            ["10:15 AM", "04:45 PM", "12:00 PM", "11:00 PM", "02:30 AM"],
            ["07:20 PM", "12:55 AM", "06:40 AM", "03:05 PM", "01:15 PM"],
        ],
        "result": [
            ["12:00 AM", "09:00 AM", "11:30 AM", "12:45 PM", "01:00 PM"],
            ["02:30 AM", "10:15 AM", "12:00 PM", "04:45 PM", "11:00 PM"],
            ["12:55 AM", "06:40 AM", "01:15 PM", "03:05 PM", "07:20 PM"],
        ],
    },
    {
        "que_number": 16,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            ["20-01-2024", "12/15/2023", "2022.05.30"],
            ["03/25/2021", "2020.12.31", "15-04-2022"],
            ["07/30/2020", "2025.01.01", "18-03-2023"],
        ],
        "result": [
            ["2022/05/30", "2023/12/15", "2024/01/20"],
            ["2020/12/31", "2021/03/25", "2022/04/15"],
            ["2020/07/30", "2023/03/18", "2025/01/01"],
        ],
    },
    {
        "que_number": 17,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            {
                "월": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
                "화": ["2024-01-02", "2024-01-09", "2024-01-16"],
                "수": ["2024-01-03", "2024-01-10"],
                "목": ["2024-01-04", "2024-01-11", "2024-01-18"],
                "금": ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"],
            },
            {
                "월": ["2024-02-05", "2024-02-12", "2024-02-19"],
                "화": ["2024-02-06", "2024-02-13", "2024-02-20"],
                "수": ["2024-02-07", "2024-02-14", "2024-02-21"],
                "목": ["2024-02-01", "2024-02-08", "2024-02-15", "2024-02-22"],
                "금": ["2024-02-02", "2024-02-09", "2024-02-16", "2024-02-23"],
            },
            {},
        ],
        "result": [
            ["24-01-26 금", "24-01-22 월", "24-01-19 금"],
            ["24-02-23 금", "24-02-22 목", "24-02-21 수"],
            [],
        ],
    },
    {
        "que_number": 18,
        "lv": 0,
        "kinds": "분석",
        "testcase": [
            {
                "2024-01-01": 15,
                "2024-01-02": 17,
                "2024-01-03": 16,
                "2024-01-04": 20,
                "2024-01-05": 19,
                "2024-01-06": 21,
                "2024-01-07": 18,
            },
            {
                "2024-02-01": 10,
                "2024-02-02": 12,
                "2024-02-03": 9,
                "2024-02-04": 15,
                "2024-02-05": 14,
                "2024-02-06": 13,
                "2024-02-07": 16,
            },
            {
                "2024-03-01": 18,
                "2024-03-02": 17,
                "2024-03-03": 19,
                "2024-03-04": 20,
                "2024-03-05": 21,
                "2024-03-06": 22,
                "2024-03-07": 20,
            },
            {
                "2024-04-01": 22,
                "2024-04-02": 24,
                "2024-04-03": 23,
                "2024-04-04": 22,
                "2024-04-05": 25,
                "2024-04-06": 24,
                "2024-04-07": 23,
            },
            {
                "2024-05-01": 20,
                "2024-05-02": 19,
                "2024-05-03": 21,
                "2024-05-04": 23,
                "2024-05-05": 22,
                "2024-05-06": 24,
                "2024-05-07": 23,
            },
        ],
        "result": [
            ["24-01-06: 21", "24-01-04: 20", "24-01-05: 19"],
            ["24-02-07: 16", "24-02-04: 15", "24-02-05: 14"],
            ["24-03-06: 22", "24-03-05: 21", "24-03-04: 20"],
            ["24-04-05: 25", "24-04-02: 24", "24-04-06: 24"],
            ["24-05-06: 24", "24-05-04: 23", "24-05-07: 23"],
        ],
    },
    {
        "que_number": 19,
        "lv": 0,
        "kinds": "타입 확인",
        "testcase": [
            [123, 4.56, "hello", [1, 2, 3], (4, 5), {"a": 1, "b": 2}],
            [True, 0, 3.14, "world", [3, 4, 5], (6, 7, 8), {"x": 3, "y": 4}],
            [[1, 2], ["a", "b", "c"], (12, 34), 567, 8.91, "Python"],
            ["text", 1001, 2.718, (9, 10, 11), [6, 7, 8], {"key": "value"}],
            [False, 2024, (13, 14, 15), "data", 5.67, [9, 10]],
        ],
        "result": [
            ["int", "float", "str", "list", "tuple", "dict"],
            ["bool", "int", "float", "str", "list", "tuple", "dict"],
            ["list", "list", "tuple", "int", "float", "str"],
            ["str", "int", "float", "tuple", "list", "dict"],
            ["bool", "int", "tuple", "str", "float", "list"],
        ],
    },
    {
        "que_number": 20,
        "lv": 0,
        "kinds": "타입 확인",
        "testcase": [
            [("list", [1, 2, 3]), ("int", 4), ("str", "hello")],
            [("str", 100), ("dict", {"a": 1}), ("tuple", (1, 2, 3))],
            [("float", 3.14), ("list", "not a list"), ("dict", {"key": "value"})],
        ],
        "result": [True, False, False],
    },
    {
        "que_number": 21,
        "lv": 0,
        "kinds": "탐색",
        "testcase": [
            [[1, 3, 5, 7, 9], 5],
            [[2, 4, 6, 8, 10], 7],
            [[10, 20, 30, 40, 50], 30],
            [[15, 25, 35, 45], 60],
            [[100, 200, 300, 400, 500, 600], 400],
        ],
        "result": [2, False, 2, False, 3],
    },
    {
        "que_number": 22,
        "lv": 0,
        "kinds": "탐색",
        "testcase": [
            ["hello", "e"],
            ["world", "a"],
            ["programming", "g"],
            ["data", "d"],
            ["science", "z"],
        ],
        "result": [1, False, 3, 0, False],
    },
    {
        "que_number": 23,
        "lv": 0,
        "kinds": "탐색",
        "testcase": [
            [[[1, 3, 5], [7, 9, 11], [13, 15, 17]], 7],
            [[[2, 4, 6], [8, 10, 12], [14, 16, 18]], 13],
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 5],
            [[[10, 11, 12], [13, 14, 15], [16, 17, 18]], 16],
            [[[3, 6, 9], [12, 15, 18], [21, 24, 27]], 22],
        ],
        "result": [True, False, True, True, False],
    },
    {
        "que_number": 24,
        "lv": 0,
        "kinds": "탐색, 동적 프로그래밍, 카데인 알고리즘",
        "testcase": [
            [1, -2, 3, 4, -1, 2, 1, -5, 4],
            [-2, -3, 4, -1, -2, 1, 5, -3],
            [-1, -2, -3, -4],
            [2, 3, -2, 5, -3],
            [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        ],
        "result": [9, 7, -1, 8, 6],
    },
    {
        "que_number": 25,
        "lv": 0,
        "kinds": "탐색, 에라토스테네스의 체",
        "testcase": [10, 5, 20, 100, 50],
        "result": [4, 3, 8, 25, 15],
    },
    {
        "que_number": 26,
        "lv": 0,
        "kinds": "탐색",
        "testcase": [
            [[1, 3, 2, 6, -1, 4, 1, 8, 2], 5],
            [[4, 2, 1, 7, 8, 1, 2, 8, 1, 0], 3],
            [[-2, -1, -3, 4, -1, 2, 1, -5, 4], 4],
            [[2, 3, 4, 1, 5], 2],
            [[1, -1, 3, -2, 5, 4, -3, 2, 1, -4], 4],
        ],
        "result": [18, 16, 6, 7, 10],
    },
    {
        "que_number": 27,
        "lv": 0,
        "kinds": "탐색",
        "testcase": [
            [[2, 1, 5, 2, 3, 2], 7],
            [[1, 1, 1, 1, 1, 1, 1], 11],
            [[3, 4, 1, 1, 6], 8],
            [[2, 3, 1, 2, 4, 3], 7],
            [[1, 4, 4], 8],
        ],
        "result": [2, 0, 3, 2, 2],
    },
    {
        "que_number": 28,
        "lv": 0,
        "kinds": "탐색",
        "testcase": [
            [[1, 2, 3, 4, 5], 10],
            [[2, 3, 5, 8, 12], 11],
            [[-1, 2, 3, 5, 8], 6],
            [[1, 3, 4, 7, 10], 8],
            [[-3, -1, 1, 2, 4], 0],
        ],
        "result": [9, 11, 7, 8, 0],
    },
    {
        "que_number": 29,
        "lv": 0,
        "kinds": "비트 조작",
        "testcase": [
            [4, 1, 2, 1, 2],
            [2, 2, 1],
            [7, 3, 5, 4, 5, 3, 4],
            [9, 1, 2, 1, 2, 9, 3],
            [10, 10, 11, 11, 12],
        ],
        "result": [4, 1, 7, 3, 12],
    },
    {
        "que_number": 30,
        "lv": 0,
        "kinds": "비트 조작",
        "testcase": [5, 9, 15, 0, 8],
        "result": ["ABA", "ABBA", "AAAA", "B", "ABBB"],
    },
    {
        "que_number": 31,
        "lv": 0,
        "kinds": "비트 조작",
        "testcase": [5, 9, 0, 15, 1023],
        "result": [1018, 1014, 1023, 1008, 0],
    },
    {
        "que_number": 32,
        "lv": 0,
        "kinds": "비트 조작",
        "testcase": [[12, 5, 6], [7, 14, 3], [1, 1, 1], [0, 0, 0], [15, 7, 3]],
        "result": [(4, 15), (2, 15), (1, 1), (0, 0), (3, 15)],
    },
    {
        "que_number": 33,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": [
            "user@example.com",
            "user.name@domain.co",
            "user@domain",
            "@example.com",
            "example@.com",
        ],
        "result": [True, True, False, False, False],
    },
    {
        "que_number": 34,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": [
            "The event will happen on 2024-01-20.",
            "Dates: 2023-12-31, 2024-01-01, and 2024-02-28.",
            "No dates here.",
        ],
        "result": [[(2024, 1, 20)], [(2023, 12, 31), (2024, 1, 1), (2024, 2, 28)], []],
    },
    {
        "que_number": 35,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": [
            "<p>Hello, <b>World</b>!</p>",
            "<div><i>Sample</i> Text</div>",
            "No tags here.",
        ],
        "result": ["Hello, World!", "Sample Text", "No tags here."],
    },
    {
        "que_number": 36,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": [
            "Contact me at 01012345678.",
            "My number is 03112345678. Call me!",
            "No numbers here.",
        ],
        "result": [
            "Contact me at 010-1234-5678.",
            "My number is 031-1234-5678. Call me!",
            "No numbers here.",
        ],
    },
    {
        "que_number": 37,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": ["[08:55:45] 사용자 로그인", "[13:01:02] 데이터베이스 접근", "[23:59:59] 시스템 종료"],
        "result": [
            {"time": "08:55:45", "message": "사용자 로그인"},
            {"time": "13:01:02", "message": "데이터베이스 접근"},
            {"time": "23:59:59", "message": "시스템 종료"},
        ],
    },
    {
        "que_number": 38,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": [
            "http://www.example.com",
            "https://www.example.com/path/to/resource?user=abc&lang=en",
            "ftp://ftp.example.com/folder?page=1",
        ],
        "result": [
            {"protocol": "http", "domain": "www.example.com", "path": "", "query": ""},
            {
                "protocol": "https",
                "domain": "www.example.com",
                "path": "/path/to/resource",
                "query": "user=abc&lang=en",
            },
            {
                "protocol": "ftp",
                "domain": "ftp.example.com",
                "path": "/folder",
                "query": "page=1",
            },
        ],
    },
    {
        "que_number": 39,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": [
            "example/document.pdf",
            "https://www.example.com/image.jpg",
            "example/folder",
            "archive.tar.gz",
        ],
        "result": ["pdf", "jpg", "", "gz"],
    },
    {
        "que_number": 40,
        "lv": 0,
        "kinds": "정규표현식",
        "testcase": ["가격은 3,500원입니다.", "오늘은 2023년 3월 5일", "문자만 있는 경우"],
        "result": ["3500", "202335", ""],
    },
    {
        "que_number": 41,
        "lv": 0,
        "kinds": "데이터 구조",
        "testcase": ["{[()()]}", "[(])", "({[]})", "({[}])"],
        "result": [True, False, True, False],
    },
    {
        "que_number": 42,
        "lv": 0,
        "kinds": "데이터 구조",
        "testcase": [
            {"size": 3, "requests": ["A", "B", "C", "D"]},
            {"size": 2, "requests": ["X", "Y", "Z"]},
            {"size": 1, "requests": ["P", "Q", "R"]},
        ],
        "result": [["B", "C", "D"], ["Y", "Z"], ["R"]],
    },
    {
        "que_number": 43,
        "lv": 0,
        "kinds": "데이터 구조",
        "testcase": [
            {"size": 3, "pages": ["page1", "page2", "page3", "page2", "page4"]},
            {"size": 2, "pages": ["pageA", "pageB", "pageC"]},
            {"size": 4, "pages": ["pageX", "pageY", "pageZ", "pageX", "pageW"]},
        ],
        "result": [
            ["page3", "page2", "page4"],
            ["pageB", "pageC"],
            ["pageY", "pageZ", "pageX", "pageW"],
        ],
    },
    {
        "que_number": 44,
        "lv": 0,
        "kinds": "데이터 구조",
        "testcase": ["Hello world, hello", "The quick brown fox", "One one one two"],
        "result": [
            {"hello": 2, "world": 1},
            {"the": 1, "quick": 1, "brown": 1, "fox": 1},
            {"one": 3, "two": 1}
        ],
    },
    {
        "que_number": 45,
        "lv": 0,
        "kinds": "데이터 구조",
        "testcase": [
            {"queue1": [1, 2, 1, 2], "queue2": [1, 10, 1, 2]},
            {"queue1": [1, 1, 1, 9], "queue2": [1, 1, 5, 3]},
            {"queue1": [1, 1, 1], "queue2": [10]}
        ],
        "result": [
            7,
            1,
            -1
        ],
    },
    {
        "que_number": 46,
        "lv": 0,
        "kinds": "데이터 구조",
        "testcase": [
            {"size": 3, "commands": ["insert 1", "insert 2", "insert 3", "insert 4", "search 3", "delete", "search 3"]},
            {"size": 2, "commands": ["insert A", "insert B", "insert C", "search B"]},
            {"size": 4, "commands": ["insert X", "delete", "search X"]}
        ],
        "result": [
            [None, None, None, None, True, None, True],
            [None, None, None, True],
            [None, None, False]
        ],
    },
]
