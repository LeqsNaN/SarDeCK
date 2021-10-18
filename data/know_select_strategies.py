# 这里完成3种知识选择，分别是通过大多数情感，少数情感，文本相似度, 通过情感相反
# three types of knowledge selection strategies: major, minor, contrast
import string


path = "SemEval/" # Ghosh, Ptacek, SARC_main SARC_politics SemEval
class SentiWordNet():
    def __init__(self, netpath):
        self.netpath = netpath
        self.dictionary = {}

    def infoextract(self):
        try:
            f = open(self.netpath, "r")
        except IOError:
            print("failed to open file!")
            exit()
        print('start extracting.......')

    # Example line:
    # POS     ID     PosS  NegS SynsetTerm#sensenumber Desc
    # a   00009618  0.5    0.25  spartan#4 austere#3 ascetical#2  ……
        for sor in f.readlines():
            if sor.strip().startswith("#"):
                pass
            else:
                data = sor.split("\t")
                if len(data) != 6:
                    print('invalid data')
                    break
                synsetScore = float(data[2]) - float(data[3])  # // Calculate synset score as score = PosS - NegS
                synTermsSplit = data[4].split(" ")  # word#id
                # ["dorsal#2", "abaxial#1"]
                for w in synTermsSplit:
                    synTermAndRank = w.split("#")
                    synTerm = synTermAndRank[0]
                    self.dictionary[synTerm] = self.dictionary.get(synTerm,0)+synsetScore

    def getscore(self, word):
        return self.dictionary.get(word,0)


def get_major_sent_know():
    netpath = "data_source/SentiWordNet_3.0.0.txt"
    know_file = path+"common_know.txt"
    output_file = path+"major_sent_know.txt"
    swn = SentiWordNet(netpath)
    swn.infoextract()
    f = open(know_file, "r")
    f_o = open(output_file ,"w")
    sep = " ==sep== "
    for num, sent in enumerate(f.readlines()):
        temp = sent.split(" ==sep== ")
        id_know = {}
        id_score = {}
        total_score = 0
        for i, know in enumerate(temp[2:-1]):
            id_know[i] = know.strip()
            id_score[i] = 0
            for word in id_know[i].split():
                id_score[i]+=float(swn.getscore(word))
            total_score+=id_score[i]
        content = []
        if total_score == 0:
            for key in id_know.keys():
                content.append(id_know[key])
        elif total_score>0:
            for key in id_know.keys():
                if id_score[key]>=0:
                    content.append(id_know[key])
        else:
            for key in id_know.keys():
                if id_score[key]<=0:
                    content.append(id_know[key])
        res_record = temp[0]+ sep + temp[1] + sep + sep.join(content) + sep+temp[-1]
        f_o.write(res_record)
    f.close()
    f_o.close()
# get_major_sent_know()

def get_minor_sent_know():
    netpath = "data_source/SentiWordNet_3.0.0.txt"
    know_file = path+"common_know.txt"
    output_file = path+"minor_sent_know.txt"
    swn = SentiWordNet(netpath)
    swn.infoextract()
    f = open(know_file, "r")
    f_o = open(output_file ,"w")
    sep = " ==sep== "
    for num, sent in enumerate(f.readlines()):
        temp = sent.split(" ==sep== ")
        id_know = {}
        id_score = {}
        total_score = 0
        for i, know in enumerate(temp[2:-1]):
            id_know[i] = know.strip()
            id_score[i] = 0
            for word in id_know[i].split():
                id_score[i]+=float(swn.getscore(word))
            total_score+=id_score[i]
        content = []
        if total_score == 0:
            for key in id_know.keys():
                content.append(id_know[key])
        elif total_score > 0:
            for key in id_know.keys():
                if id_score[key] <= 0:
                    content.append(id_know[key])
        else:
            for key in id_know.keys():
                if id_score[key] >= 0:
                    content.append(id_know[key])
        res_record = temp[0]+ sep + temp[1] + sep + sep.join(content) + sep+temp[-1]
        f_o.write(res_record)
    f.close()
    f_o.close()
# get_minor_sent_know()

def get_contrast_sent_know():
    netpath = "data_source/SentiWordNet_3.0.0.txt"
    know_file = path + "common_know.txt"
    output_file = path + "contrast_sent_know.txt"
    swn = SentiWordNet(netpath)
    swn.infoextract()
    f = open(know_file, "r")
    f_o = open(output_file, "w")
    sep = " ==sep== "
    for num, sent in enumerate(f.readlines()):
        temp = sent.split(" ==sep== ")
        id_know = {}
        id_score = {}
        text_score = 0
        for text in temp[1].split():
            word = ''.join(c for c in text if c not in string.punctuation)
            text_score+= float(swn.getscore(word))
        for i, know in enumerate(temp[2:-1]):
            id_know[i] = know.strip()
            id_score[i] = 0
            for word in id_know[i].split():
                id_score[i]+=float(swn.getscore(word))
        content = []
        if text_score == 0:
            for key in id_know.keys():
                content.append(id_know[key])
        elif text_score > 0:
            for key in id_know.keys():
                if id_score[key] <= 0:
                    content.append(id_know[key])
        else:
            for key in id_know.keys():
                if id_score[key] >= 0:
                    content.append(id_know[key])
        res_record = temp[0]+ sep + temp[1] + sep + sep.join(content) + sep+temp[-1]
        f_o.write(res_record)
    f.close()
    f_o.close()
# get_contrast_sent_know()


