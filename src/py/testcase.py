# PAGE_NAME으로 바로 호출 가능하도록 0번째 값을 비워줌
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
            "C",
        ]
    },
    {
        "que_number": 10,
        "lv": 0,
        "kinds": "정렬",
        "testcase": [
            [
                ['Licat', 98, 92, 85, 97], 
                ['Mura', 95, 32, 51, 30], 
                ['Binky', 98, 98, 51, 32],
            [
                ['Gray', 98, 92, 85, 97], 
                ['Gom', 98, 30, 21, 60], 
                ['Allosa', 98, 90, 99, 98],
            ],
            [
                ['A', 10, 15, 20, 25], 
                ['B', 30, 35, 41, 10], 
                ['C', 18, 30, 29, 18],
            ],
        ],
        "result": [
            ['Licat'],
            ['Allosa', 'Gray'],
            [],
        ],
    },
]
