import random
sarcasm_data = []
normal_data = []

dict = {}

#
# f = open("train-balanced-sarcasm.csv",'r', encoding='utf-8')
# for line in f:
#     temp = line.strip().split(",")
#     res = []
#     dict[temp[3]] = dict.get(temp[3],0)+1
#     # if temp[3] == "politics":
#     #     if temp[0]=="0":
#     #         res.append(temp[1].strip())
#     #         res.append("0")
#     #         normal_data.append(res)
#     #     elif temp[0] == "1":
#     #         res.append(temp[1].strip())
#     #         res.append("1")
#     #         sarcasm_data.append(res)
# f.close()
# temp = sorted(dict, key=lambda x:dict[x], reverse=True)[:20]
# for i in temp:
#     print(i,dict[i])

# split data
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
