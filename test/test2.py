import numpy as np

all_feature = {'BL=B': 0, 'BL=M': 1, 'BL=E': 2, 'BL=S': 3, 'BL=_BL_': 4}  # 特征取值集合
sentence_feature_dict = {}  # 记录每个句子中每个字的特征，key是句子，value是特征list
sentence_tag_dict = {}  # 每个字的类标，，key是句子，value是类标list
all_sentences = []  # 所有句子
START_CHAR = '\1'
END_CHAR = '\2'

count = 0


def handle_feature(feature, char_feature, all_feature):
    feature_id = all_feature[feature] if feature in all_feature else len(all_feature)
    all_feature[feature] = feature_id
    char_feature.append(feature_id)


# weights = np.ones(())
data_f = open('/Users/a1/PycharmProjects/AI-NLP-Tutorial/segmentation/machine-learning/data/RenMinData.txt', 'r', encoding='utf-8')
for line in data_f.readlines():
    count += 1
    line = line.strip()
    # 打标签
    words = line.split(' ')
    all_sentences.append(''.join(words))
    tag_list = []
    for word in words:
        if len(word) == 1:
            tag_list.append(all_feature['BL=S'])
        else:
            tag_list.append(all_feature['BL=B'])
            for w in word[1:len(word) - 1]:  # 中间字
                tag_list.append(all_feature['BL=M'])
            tag_list.append(all_feature['BL=E'])
    sentence_tag_dict[''.join(words)] = tag_list

    # 获取特征
    sentence = line.replace(' ', '')
    sentence_feature = []
    for i, char in enumerate(sentence):
        char_feature = []
        # 前2
        pre2 = sentence[i - 2] if i >= 2 else START_CHAR
        # 前1
        pre1 = sentence[i - 1] if i >= 1 else START_CHAR
        # 当前
        cur = char
        # 后1
        next1 = sentence[i + 1] if i < len(sentence) - 1 else END_CHAR
        # 后2
        next2 = sentence[i + 2] if i < len(sentence) - 2 else END_CHAR

        # unigrams
        one = pre1 + '1'
        handle_feature(one, char_feature, all_feature)

        two = cur + '2'
        handle_feature(two, char_feature, all_feature)

        three = next1 + '3'
        handle_feature(three, char_feature, all_feature)

        # bigrams
        four = pre2 + '/' + pre1 + '4'
        handle_feature(four, char_feature, all_feature)

        five = pre1 + '/' + cur + '5'
        handle_feature(five, char_feature, all_feature)

        six = cur + '/' + next1 + '6'
        handle_feature(six, char_feature, all_feature)

        seven = next1 + '/' + next2 + '7'
        handle_feature(seven, char_feature, all_feature)

        sentence_feature.append(char_feature)
        # print('char_feature',char_feature)
    sentence_feature_dict[''.join(words)] = sentence_feature
print('数据集处理完成！')

import random
import time
import numpy as np

tag_list = [0, 1, 2, 3]  # 标签id
hidden_node_num = len(tag_list)  # 隐状态个数（标签个数）
# 参数
global W
# W = np.random.randn(hidden_node_num, len(all_feature))
W = np.zeros((hidden_node_num, len(all_feature)))


def viterbi_decode(sentence_feature, W):
    path_prob = []  # 路径和该路径的概率
    best_path = []  # 最好的路径，也就是最好的tag
    for i, char_feature in enumerate(sentence_feature):

        if i == 0:
            char_feature_temp = char_feature + [4]  # 前一个字的标签
            Z = [sum(W[tag][char_feature_temp]) for tag in tag_list]  # 每个tag的概率【因为是第一个字所以只用到发射概率和初始概率】
            # 第一个tag是当前字的标签，第二个tag是上一个字的标签
            path_prob.append({tag: (tag, prob) for tag, prob in enumerate(Z)})
        else:
            Z = [sum(W[tag][char_feature]) for tag in tag_list]  # 每个tag的概率
            res = {}
            for j, prob in enumerate(Z):
                # 加上上一个字的标签分别为0，1，2，3时的概率，然后求最大【发射概率、转移概率、初始概率】
                max_prob, pre_tag = max(
                    [(prob + W[j][pre_tag] + value[1], pre_tag) for pre_tag, value in path_prob[i - 1].items()])
                res[j] = (pre_tag, max_prob)
            path_prob.append(res)
    # print('path_prob:',path_prob)
    # 确定最后一个字的tag，然后回溯，确定路径（确定每个字的tag）
    last_tag = -1
    max_prob = -100000
    for tag in tag_list:
        prob = path_prob[-1][tag][1]
        if max_prob < prob:
            last_tag = tag
            max_prob = prob
    # last_tag = sorted(, key=lambda x: x[1][1], reverse = True)[0][0]
    # print('last_tag:', last_tag)
    # last_tag = np.argmax([pp[1][1] for pp in path_prob[-1]])
    best_path.append(last_tag)
    for pp in reversed(path_prob):
        best_path.append(pp[last_tag][0])
        last_tag = pp[last_tag][0]
    best_path.pop()
    best_path.reverse()
    # print('best_path:', best_path)
    return best_path


def precision_and_recall(predict_sentence_tags):
    """
    训练集准确率： 0.7824
    验证集准确率： 0.7478856222311719
    """
    correct_count = 0
    for sentence, predict_sentence_tag in predict_sentence_tags:
        if predict_sentence_tag == sentence_tag_dict[sentence]:
            correct_count += 1
    return correct_count / len(predict_sentence_tags)


def precision_and_recall2(predict_sentence_tags):
    """
    训练集准确率： 0.964133612502998
    验证集准确率： 0.9558979114692859
    """
    correct_count = 0
    total_count = 0
    for sentence, predict_sentence_tag in predict_sentence_tags:
        for i, tag in enumerate(predict_sentence_tag):
            total_count += 1
            if tag == sentence_tag_dict[sentence][i]:
                correct_count += 1
    return correct_count / total_count


# 训练
def train(all_sentences, maxIteration, ratio=0.7):
    global W
    end = int(np.floor(ratio * len(all_sentences)))
    print(end)
    x_train = all_sentences[:end]
    x_val = all_sentences[end:]
    for iter in range(maxIteration):
        random.shuffle(x_train)  # 打乱
        start = time.time()
        for i, sentence in enumerate(x_train):
            sentence_feature = sentence_feature_dict[sentence]
            #             print('i--', i)
            if i % 10000 == 0:
                #                 print('10000次耗时：', time.time()-start)
                start = time.time()
                print('i--', i)
            #             start = time.time()
            predict_tag = viterbi_decode(sentence_feature, W)
            #             print('viterbi_decode耗时：', time.time()-start)
            actual_tag = sentence_tag_dict[sentence]
            #             print('predict_tag:', predict_tag)
            #             print('actual_tag:', actual_tag)
            if predict_tag == actual_tag:
                # print('true')
                continue
            else:
                # 更新权重：每个特征上的实际标签权重增加(+1)，其他标签权重减少(-1)
                #                 start = time.time()
                for j, char_feature in enumerate(sentence_feature):
                    #                     print('predict_tag:', predict_tag)
                    #                     print('actual_tag:', actual_tag)
                    if predict_tag[j] == actual_tag[j]:
                        continue
                    #                     print('char_feature:',char_feature)
                    for tag in tag_list:
                        if tag == actual_tag[j]:
                            # print('W[tag][char_feature] == ', W[tag][char_feature])
                            W[tag][char_feature] += 1
                            if j == 0:  # 第一个字
                                W[tag][4] += 1
                            else:
                                W[tag][actual_tag[j - 1]] += 1

                            # print('W[tag][char_feature] ++1 == ', W[tag][char_feature])
                        elif tag == predict_tag[j]:
                            W[tag][char_feature] -= 1
                            if j == 0:  # 第一个字
                                W[tag][4] -= 1
                            else:
                                W[tag][predict_tag[j - 1]] -= 1
        #                 print('W耗时：', time.time()-start)

        # 本轮次训练集和验证集上的结果
        train_predict_sentence_tags = [(sentence, viterbi_decode(sentence_feature_dict[sentence], W)) for sentence in
                                       x_train[:1000]]
        print('训练集准确率：', precision_and_recall2(train_predict_sentence_tags))
        val_predict_sentence_tags = [(sentence, viterbi_decode(sentence_feature_dict[sentence], W)) for sentence in
                                     x_val]
        print('验证集准确率：', precision_and_recall2(val_predict_sentence_tags))


print('总共句子数：', len(all_sentences))
train(all_sentences, 5, ratio=0.7)