import codecs
from sklearn.model_selection import train_test_split  # 进行训练集和测试集划分
import pickle  # 进行参数保存

INPUT_DATA = "./machine-learning/data/RenMinData.txt"  # 数据集
SAVE_PATH = "./machine-learning/model/datasave.pkl"  # 保存路径
id2tag = ['B', 'M', 'E', 'S']  # B：分词头部 M：分词词中 E：分词词尾 S：独立成词 id与状态值
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}  # 状态值对应的id
word2id = {}  # 每个汉字对应的id
id2word = []  # 每个id对应的汉字


def getList(input_str):
    '''
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    '''
    outpout_str = []
    if len(input_str) == 1:  # 长度为1 单个字分词
        outpout_str.append(tag2id['S'])
    elif len(input_str) == 2:  # 长度为2 两个字分词，BE
        outpout_str = [tag2id['B'], tag2id['E']]
    else:  # 长度>=3 多个字分词 中间加length-2个M 首尾+BE
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        outpout_str.append(tag2id['B'])
        outpout_str.extend(M_list)
        outpout_str.append(tag2id['E'])
    return outpout_str


def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    x_data = []  # 观测值序列集合
    y_data = []  # 状态值序列集合
    wordnum = 0
    line_num = 0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:  # 对每一个sentence
            line_num = line_num + 1
            line = line.strip()
            if not line: continue
            line_x = []
            for i in range(len(line)):
                if line[i] == " ": continue
                if (line[i] in id2word):  # word与id对应进行记录
                    line_x.append(word2id[line[i]])
                else:
                    id2word.append(line[i])
                    word2id[line[i]] = wordnum
                    line_x.append(wordnum)
                    wordnum = wordnum + 1
            x_data.append(line_x)

            lineArr = line.split(" ")
            line_y = []
            for item in lineArr:  # 对每一个分词进行状态值转换
                line_y.extend(getList(item))
            y_data.append(line_y)

    print(x_data[0])
    print([id2word[i] for i in x_data[0]])
    print(y_data[0])
    print([id2tag[i] for i in y_data[0]])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=43)  # 分为训练集和测试集
    with open(SAVE_PATH, 'wb') as outp:  # 保存
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)

Trans = {}  #trans
Emit = {}  #emit
Count_dic = {}
Start = {}  #start

def init():
    '''
    参数初始化
    Trans = {}  # 状态转移矩阵
    Emit = {}  # 观测概率矩阵
    Count_dic = {} # 每个状态的数量计数
    Start = {}  # 初始概率矩阵
    '''
    for tag in tag2id:
        Trans[tag2id[tag]] = {}
        for tag2 in tag2id:
            Trans[tag2id[tag]][tag2id[tag2]] = 0.0
    for tag in tag2id:
        Start[tag2id[tag]] = 0.0
        Emit[tag2id[tag]] = {}
        Count_dic[tag2id[tag]] = 0

    print('完成')


def train():
    '''
    根据输入的训练集进行各个数组的填充
    :return:
    '''
    with open('../data/datasave.pkl', 'rb') as inp:
        '''
        读取数据处理结果
        '''
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    for sentence, tags in zip(x_train, y_train):
        for i in range(len(tags)):
            if i == 0:
                Start[tags[0]] += 1
                Count_dic[tags[0]] += 1
            else:
                Trans[tags[i - 1]][tags[i]] += 1
                Count_dic[tags[i]] += 1
                if sentence[i] not in Emit[tags[i]]:
                    Emit[tags[i]][sentence[i]] = 0.0
                else:
                    Emit[tags[i]][sentence[i]] += 1

    for tag in Start:
        Start[tag] = Start[tag] * 1.0 / len(x_train)
    for tag in Trans:
        for tag1 in Trans[tag]:
            Trans[tag][tag1] = Trans[tag][tag1] / Count_dic[tag]

    for tag in Emit:
        for word in Emit[tag]:
            Emit[tag][word] = Emit[tag][word] / Count_dic[tag]
    print(Start)
    print(Trans)

if __name__ == "__main__":
    # handle_data()
    init()

