# PAGE_NAME으로 바로 호출 가능하도록 0번째 값을 비워줌
# black formatter 적용하면 안됨(마지막에 콤마 생기는 것때문에 error 발생)
testcase_and_result = [
    {"que_number": 0, "lv": "", "kinds": "", "testcase": "", "result": ""},
    {
        "que_number": 1,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [
            [1, 2, 3, 4, 5],
            [],
            [1, 3, 5, 7, 9],
            [2, 3, 4],
            [2, 4, 6, 8]
        ],
        "result": [9, 0, 25, 3, 0]
    },
    {
        "que_number": 2,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [
            [1, 2, 3, 4, 5],
            [],
            [1, 3, 5, 7],
            [2, 3, 4],
            [2, 4, 6, 8]
        ],
        "result": [120, 0, 105, 24, 384]
    },
    {
        "que_number": 3,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [
            [1, 2, 3, 4, 5],
            [],
            [1, 3, 5, 7],
            [2, 3, 4],
            [2, 4, 6, 8]
        ],
        "result": [7, 0, 8, 6, 14]
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
            ["쿠키 1개", "쿠키 1개", "쿠키 1개"]
        ],
        "result": [14, 15, 18, 22, 6]
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
            [1, 11, 1, 11]
        ],
        "result": [1, 10, 1, 0, 6]
    },
    {
        "que_number": 6,
        "lv": 0,
        "kinds": "요구사항 구현",
        "testcase": [
            "1hel2lo3",
            "1q2w3e4r",
            "",
            "hello",
            "1234"
        ],
        "result": [6, 10, 0, 0, 10]
    },
    {
        "que_number": 7,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [["A", 1], ["B", 2], ["C", 3]],
            [["A", 3], ["B", 2], ["C", 1]],
            [["A", 1], ["B", 3], ["C", 2]],
            [["A", 1], ["B", 3], ["C", 2], ["D", 5], ["E", 4]]
        ],
        "result": [
            ["A", "B", "C"],
            ["C", "B", "A"],
            ["A", "C", "B"],
            ["A", "C", "B", "E", "D"]
        ]
    },
    {
        "que_number": 8,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [1, 3, 5, 7, 8, 10, 12],
            [10, 20, 30, 35, 45, 55],
            [4, 8, 12, 13, 20, 24],
            [19, 20, 30, 40, 50, 60, 70]
        ],
        "result": [
            [7, 8],
            [30, 35],
            [12, 13],
            [19, 20]
        ]
    },
    {
        "que_number": 9,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                {"이름": "A", "국": 30, "영": 38, "수": 67},
                {"이름": "B", "국": 95, "영": 21, "수": 98},
                {"이름": "C", "국": 92, "영": 33, "수": 32}
            ],
            [
                {"이름": "A", "국": 67, "영": 67, "수": 81},
                {"이름": "B", "국": 82, "영": 32, "수": 98},
                {"이름": "C", "국": 95, "영": 11, "수": 99}
            ],
            [
                {"이름": "A", "국": 33, "영": 64, "수": 37},
                {"이름": "B", "국": 92, "영": 38, "수": 89},
                {"이름": "C", "국": 31, "영": 98, "수": 100}
            ]
        ],
        "result": [
            "B",
            "C",
            "C"
        ]
    },
    {
        "que_number": 10,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ["Licat", 98, 92, 85, 97],
                ["Mura", 95, 32, 51, 30], 
                ["Binky", 98, 98, 51, 32]
            ],
            [
                ["Gray", 98, 92, 85, 97], 
                ["Gom", 98, 30, 21, 60], 
                ["Allosa", 98, 90, 99, 98]
            ],
            [
                ["A", 10, 15, 20, 25], 
                ["B", 30, 35, 41, 10], 
                ["C", 18, 30, 29, 18]
            ]
        ],
        "result": [
            ["Licat"],
            ["Allosa", "Gray"],
            []
        ]
    },
    {
        "que_number": 11,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [[98, 92, 85], [95, 32, 51], [98, 98, 51]],
            [[92, 85, 97], [30, 21, 60], [90, 99, 98], [0, 0, 0], [81, 80, 88, 83]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ],
        "result": [
            2,
            3,
            0
        ]
    },
    {
        "que_number": 12,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ["제주시 A동 한라산길 61", "제주시 B동 백록담길 63", "제주시 C동 사라봉길 31"],
                {"A동": 63007, "B동": 63010, "C동": 63002}
            ],
            [
                ["제주시 E동 한라산길 11", "제주시 D동 한라산길 101", "제주시 F동 한라산길 21"],
                {"E동": 63107, "D동": 63310, "F동": 63032}
            ],
            [
                ["제주시 AE동 한라산길 61", "제주시 FE동 백록담길 63", "제주시 BE동 사라봉길 31"],
                {"AE동": 63111, "FE동": 63132, "BE동": 63337}
            ]
        ],
        "result": [
            ["제주시 C동 사라봉길 31", "제주시 A동 한라산길 61", "제주시 B동 백록담길 63"],
            ["제주시 F동 한라산길 21", "제주시 E동 한라산길 11", "제주시 D동 한라산길 101"],
            ["제주시 AE동 한라산길 61", "제주시 FE동 백록담길 63", "제주시 BE동 사라봉길 31"]
        ]
    },
    {
        "que_number": 13,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ["Great Expectations", "Brave New World", "The Catcher in the Rye"],
                {"Great Expectations": 1861, "Brave New World": 1932, "The Catcher in the Rye": 1951}
            ],
            [
                ["To Kill a Mockingbird", "1984", "Animal Farm"],
                {"To Kill a Mockingbird": 1960, "1984": 1949, "Animal Farm": 1945}
            ],
            [
                ["The Great Gatsby", "Moby Dick", "Pride and Prejudice"],
                {"The Great Gatsby": 1925, "Moby Dick": 1851, "Pride and Prejudice": 1813}
            ]
        ],
        "result": [
            ["Great Expectations", "Brave New World", "The Catcher in the Rye"],
            ["Animal Farm", "1984", "To Kill a Mockingbird"],
            ["Pride and Prejudice", "Moby Dick", "The Great Gatsby"]
        ]
    },
    {
        "que_number": 14,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            {
                "BX32": "1984", 
                "AX21": "Moby Dick", 
                "CX14": "To Kill a Mockingbird"
            },
            {
                "ZA10": "Pride and Prejudice", 
                "YA24": "The Great Gatsby", 
                "XA33": "The Catcher in the Rye"
            },
            {
                "LD44": "Brave New World", 
                "KC55": "Great Expectations", 
                "MD66": "Animal Farm"
            }
        ],
        "result": [
            ["Moby Dick", "1984", "To Kill a Mockingbird"],
            ["The Catcher in the Rye", "The Great Gatsby", "Pride and Prejudice"],
            ["Great Expectations", "Brave New World", "Animal Farm"]
        ]
    },
    {
        "que_number": 15,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            ['01:00 PM', '11:30 AM', '12:45 PM', '09:00 AM', '12:00 AM'],
            ['10:15 AM', '04:45 PM', '12:00 PM', '11:00 PM', '02:30 AM'],
            ['07:20 PM', '12:55 AM', '06:40 AM', '03:05 PM', '01:15 PM']
        ],
        "result": [
            ['12:00 AM', '09:00 AM', '11:30 AM', '12:45 PM', '01:00 PM'],
            ['02:30 AM', '10:15 AM', '12:00 PM', '04:45 PM', '11:00 PM'],
            ['12:55 AM', '06:40 AM', '01:15 PM', '03:05 PM', '07:20 PM']
        ]
    },
    {
        "que_number": 16,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            ['20-01-2024', '12/15/2023', '2022.05.30'],
            ['03/25/2021', '2020.12.31', '15-04-2022'],
            ['07/30/2020', '2025.01.01', '18-03-2023']
        ],
        "result": [
            ['2022/05/30', '2023/12/15', '2024/01/20'],
            ['2020/12/31', '2021/03/25', '2022/04/15'],
            ['2020/07/30', '2023/03/18', '2025/01/01']
        ]
    }
]
