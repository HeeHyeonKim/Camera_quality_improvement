"""

train.csv, test.csv 파일을 읽어와
학습에 사용하기 용이하도록 txt 파일을 생성합니다.

"""

import csv

train_path = "./dataset/train.csv"
test_path = "./dataset/test.csv"

train_csv = open(train_path, 'r')
test_csv = open(test_path, 'r')

train_txt = open("./dataset/train.txt", 'w')
test_txt = open("./dataset/test.txt", 'w')

train_reader = csv.reader(train_csv)
for row in train_reader:
    line = row[0] + ", " + row[1] + ", " + row[2] + "\n"
    train_txt.write(line)

test_reader = csv.reader(test_csv)
for row in test_reader:
    line = row[0] + ", " + row[1] + ", " + row[2] + "\n"
    test_txt.write(line)