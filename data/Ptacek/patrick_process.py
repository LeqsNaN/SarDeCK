import random
# contents = []
#
#
#
# f = open("normal.tweets.tsv",'r',encoding='utf-8')
# for line in f:
#     temp = line.strip().split("\t")
#     if temp[-1] == "Not Available": continue
#     if len(temp) == 1:
#         contents[-1][0] = contents[-1][0] + " " + " ".join(temp)
#         continue
#     temp.pop(0)
#     temp.append("0")
#     contents.append(temp)
# f.close()
#
#
# f = open("sarcastic.tweets.tsv",'r',encoding='utf-8')
# for line in f:
#     temp = line.strip().split("\t")
#     if temp[-1] == "Not Available":continue
#     if len(temp)==1:
#         contents[-1][0] = contents[-1][0] +" "+ " ".join(temp)
#         continue
#     temp.pop(0)
#     temp.append("1")
#     contents.append(temp)
# f.close()
# random.shuffle(contents)
#
# f = open("data.txt", 'w')
# for content in contents:
#     f.write(content[0] + " " + content[1]+"\n")
# f.close()


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

