# 对推特数据进行处理 (preprocess the twitter data)，
# 包括小写化 (lowercasing)，
# 去http链接 (removing http links)，
# 去掉包含@的user token (removing @user token)，
# 删除包含sarcasm的关键字 (removing "sarcasm"-related hashtags)，
# 删除无关标点符号 (removing punctuations)，
# 对hashtag进行切割，基于wordninja (using wordninja to segment hashtags)
import re
import string
from collections import Counter

import wordninja
from nltk.corpus import stopwords
from textblob import Word


path = "SARC_politics/" # Ghosh, Ptacek, SARC_main



def wordBreak(text):
    res_string = []
    text = text.split()
    for token in text:
        if "#" == token[0]:
            temp = wordninja.split(token[1:])
            for i in temp:
                res_string.append(i)
        else:
            res_string.append(token)
    return " ".join(res_string)


def tweet_data_process(sfile, ofile):
    f = open(sfile, 'r', encoding='utf-8')
    punc = string.punctuation
    contents = []
    labels = []
    for line in f:
        line = line.split()
        label = line[-1]
        text = " ".join(line[:-1])
        text = text.lower()
        text = re.sub('http[s]?://\S+', '', text)
        text = [token for token in text.split() if "@" not in token]
        text = " ".join(text)
        text = text.replace("#sarcasm", "")
        text = text.replace("#Sarcasm", "")
        text = text.replace("#sarcastic", "")
        text = text.replace("#Sarcastic", "")
        if "#" in text:
            text = wordBreak(text)
        text = text.replace("#", "")
        text = text.replace("\"", "")
        res = ""
        for c in text:
            if c in punc:
                res+=" "+c
            else:
                res+=c
        text = " ".join(res.split())
        contents.append(text)
        labels.append(label)
    f.close()
    f = open(ofile, 'w')
    for id, content in enumerate(contents):
        f.write(str(id) + " ==sep== " + content + " ==sep== " + labels[id] + "\n")
    f.close()


# tweet_data_process(path+"data.txt",path+"processed_data.txt")


def sarc_data_process(sfile, ofile):
    f = open(sfile, 'r', encoding='utf-8')
    punc = string.punctuation
    contents = []
    labels = []
    for line in f:
        line = line.split()
        label = line[-1]
        text = " ".join(line[:-1])
        text = text.lower()
        text = re.sub('http[s]?://\S+', '', text)
        text = [token for token in text.split() if "@" not in token]
        text = " ".join(text)
        text = text.replace("\"", "")
        res = ""
        for c in text:
            if c in punc:
                res += " " + c
            else:
                res += c
        text = " ".join(res.split())
        contents.append(text)
        labels.append(label)
    f.close()
    f = open(ofile, 'w')
    for id, content in enumerate(contents):
        f.write(str(id) + " ==sep== " + content + " ==sep== " + labels[id] + "\n")
    f.close()


sarc_data_process(path+"data.txt",path+"processed_data.txt")

def know_data_process(sfile, ofile):
    stop = stopwords.words('english')
    f = open(sfile, 'r', encoding='utf-8')
    punc = string.punctuation
    ids = []
    contents = []
    labels = []
    for line in f:
        line = line.split(" ==sep== ")
        id = line[0]
        content = line[1]
        label = line[2]
        # deleting punctuatins
        content = " ".join([x for x in content.split() if x not in punc])
        content = " ".join([x for x in content.split() if "\'" not in x])
        res = []
        for token in content.split():
            token = "".join([c for c in token if c not in punc])
            res.append(token)
        content = " ".join(res)
        # removing stop words
        content = " ".join([x for x in content.split() if x not in stop])
        # lemmatizing
        content = " ".join([Word(word).lemmatize() for word in content.split()])
        ids.append(id)
        contents.append(content)
        labels.append(label)
    f.close()
    f = open(ofile, 'w')
    for id, content in enumerate(contents):
        f.write(ids[id] + " ==sep== " + content + " ==sep== " + labels[id])
    f.close()
know_data_process(path+"processed_data.txt", path+"know_data.txt")

def get_data_info(data_path):
    sent_max_len = 0
    avg_len = 0
    labels = []
    with open(data_path) as f:
        for line in f.readlines():
            sp = line.strip().split("==sep==")
            labels.append(sp[-1])
            avg_len += len(sp[1].split())
            if len(sp[1].split()) > sent_max_len:
                sent_max_len = len(sp[1].split())
    print("Data information: ")
    print("Total samples：", len(labels))
    print("Max sentence length：", sent_max_len)
    print("Avg sentence length：", avg_len / len(labels))
    counter = Counter(labels)
    print("Label distribution：")
    for w in counter:
        print(w, counter[w])


# get_data_info(path + "know_data.txt")