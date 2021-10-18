# def data_split(datafile):
#     f1 = open("train.txt", 'w')
#     f2 = open("dev.txt", 'w')
#     f3 = open("test.txt", 'w')
#     contents = []
#     with open(datafile) as f:
#         for idx, line in enumerate(f.readlines()):
#             contents.append(line)
#         n_train = int(51189*0.9)
#         n_val = int(51189*0.1)
#         n_test = int(51189)
#         for item in contents[:n_train]:
#             f1.write(item)
#         for item in contents[n_train:n_train+n_val]:
#             f2.write(item)
#         for item in contents[n_test:]:
#             f3.write(item)
#     f1.close()
#     f2.close()
#     f3.close()
#
# data_split("processed_data.txt")
#
import random

def data_split(datafile):
    f1 = open("train.txt", 'w')
    f2 = open("dev.txt", 'w')
    f3 = open("test.txt", 'w')
    contents = []
    with open(datafile) as f:
        for idx, line in enumerate(f.readlines()):
            contents.append(line)
        n_total = len(contents)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = int(0.1 * n_total)
        random.shuffle(contents)
        for item in contents[:n_train]:
            f1.write(item)
        for item in contents[n_train:n_train+n_val]:
            f2.write(item)
        for item in contents[n_train+n_val:n_train+n_val+n_test]:
            f3.write(item)
    f1.close()
    f2.close()
    f3.close()

data_split("processed_data.txt")

