#!/usr/bin/python
#coding=utf-8

import datetime
import math
import sys

"""类名称：sentence
   类功能：存储一个句子的组成内容，包括：所有的word、tag、以及每一个word对应单个字wordchars"""
class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

"""类名称：dataset
   类功能：数据集类，用来存储一个conll格式的文件，以数组的形式存储所有的sentence"""
class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
        self.total_word_count = 0

    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode = 'r')
        self.name = inputfile

    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        wordCount = 0
        sentenceCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\r\n' or s == '\n'):  #空白行的判断，仅仅只有换行符==空白行==新的一个句子
                sentenceCount += 1
                self.sentences.append(sen)
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')  #以\t来作为分隔符
            str_word = list_s[1].decode('utf-8')  #decode('utf8')的原因是为了更好的处理wordchars(python2.7需要这样做，3.0以上不需要)
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        self.total_word_count = wordCount  #统计word的个数
        self.total_sentence_count = len(self.sentences)  #统计sentence的个数
        print(self.name + " contains " + str(self.total_sentence_count) + " sentences")
        print(self.name + " contains " + str(self.total_word_count) + " words")

"""类名称：global_linear_model
   类功能：提供global linear model的相关功能函数：训练，评估"""
class conditional_random_field:
    def __init__(self):
        self.feature = dict()  #存储训练集得到的特征字典集合
        self.feature_keys = []
        self.feature_values = []
        self.feature_length = 0
        self.bigram_feature_id = {}
        self.tags = dict()  #存储所有的词性tag字典集合
        self.tags_length = 0
        self.actual_tags = {}
        self.g = []  #g
        self.update_times =[]  #更新的时间
        self.w = []  #权重weight
        self.train = dataset()  #训练集
        self.dev = dataset()  #开发集

        self.train.open_file("train.conll")
        #self.train.open_file("./data/train.conll")
        #self.train.open_file("./new_data/train_new.conll")
        #self.train.open_file("zy.txt")
        self.train.read_data(100)
        self.train.close_file()

        self.dev.open_file("dev.conll")
        #self.dev.open_file("./data/dev.conll")
        #self.dev.open_file("./new_data/dev_new.conll")
        self.dev.read_data(10)
        self.dev.close_file()

    """函数名称：create_bigram_feature
       函数功能：根据参数：句子sentence，位置i，左边一个词的词性ti_left_tag，构造句子当前位置的bigram 部分feature，0 <= i <= sentence_length
       函数返回：bigram feature特征，list类型"""
    def create_bigram_feature(self, ti_left_tag):
        feature = []
        feature.append("01:" + ti_left_tag)
        return feature[:]

    """函数名称：create_unigram_feature
       函数功能：根据参数：句子sentence，位置pos，构造当前位置的unigram部分特征
       函数返回：部分unigram特征，list类型"""
    def create_unigram_feature(self, sentence, i):
        position = i
        sentence_len = len(sentence.word)
        word_i = sentence.word[position]
        word_i_len = len(sentence.word[position])
        if(position == 0):
            wi_left_word = "$$"
            wi_left_word_last_c = "$$"
        else:
            wi_left_word = sentence.word[position-1]
            wi_left_word_last_c = sentence.wordchars[position-1][len(sentence.word[position-1])-1]
        if(position == sentence_len-1):
            wi_right_word = "##"
            wi_right_word_first_c = "##"
        else:
            wi_right_word = sentence.word[position+1]
            wi_right_word_first_c = sentence.wordchars[position+1][0]
        wi_last_c = sentence.wordchars[position][-1]
        wi_first_c = sentence.wordchars[position][0]
        f = []  #声明特征数组f
        f.append("02:" + word_i)
        f.append("03:" + wi_left_word)
        f.append("04:" + wi_right_word)
        f.append("05:" + word_i + '*' + wi_left_word_last_c)
        f.append("06:" + word_i + '*' + wi_right_word_first_c)
        f.append("07:" + wi_first_c)
        f.append("08:" + wi_last_c)
        for i in range(1, word_i_len - 1):
            wi_kth_c = sentence.wordchars[position][i]
            f.append("09:" + wi_kth_c)
            f.append("10:" + wi_first_c + "*" + wi_kth_c)
            f.append("11:" + wi_last_c + "*" + wi_kth_c)
        for i in range(0, word_i_len - 1):
            wi_kth_c = sentence.wordchars[position][i]
            wi_kth_next_c = sentence.wordchars[position][i + 1]
            if(wi_kth_c == wi_kth_next_c):
                f.append("13:" + wi_kth_c + "*" + "consecutive")
        if(word_i_len == 1):
            f.append("12:" + word_i + "*" + wi_left_word_last_c + "*" + wi_right_word_first_c)
        for i in range(0, word_i_len):
            if(i >= 4):
                break
            f.append("14:" + sentence.word[position][0:(i + 1)])
            f.append("15:" + sentence.word[position][-(i + 1)::])
        return f[:]

    def create_feature(self, sentence, i):
        feature = []
        if i == 0:
            ti_left_tag = "START"
        else:
            ti_left_tag = sentence.tag[i - 1]
        feature.extend(self.create_bigram_feature(ti_left_tag))
        if i == len(sentence.word):
            return feature
        feature.extend(self.create_unigram_feature(sentence, i))
        return feature

    """函数名称：create_feature_space
       函数功能：创建train对应的特征空间
       函数返回：返回dict类型的特征空间"""
    def create_feature_space(self):
        feature_index = 0
        tag_index = 0
        for s in self.train.sentences:
            sentence_length = len(s.word)
            for p in range(0, sentence_length + 1):  #句子的STOP的bigram？
                f = self.create_feature(s, p)
                for feature in f:
                    if (feature in self.feature):
                        pass
                    else:
                        self.feature[feature] = feature_index
                        feature_index += 1
                if p != sentence_length:
                    tag_p = s.tag[p]
                else:
                    tag_p = "STOP"
                if(tag_p in self.tags):
                    pass
                else:
                    self.tags[tag_p] = tag_index
                    tag_index += 1
        self.feature_length = len(self.feature)
        self.tags_length = len(self.tags)
        self.w = [0]*(self.feature_length * self.tags_length)
        self.v = [0]*(self.feature_length * self.tags_length)
        self.update_times = [0]*(self.feature_length * self.tags_length)
        self.feature_keys = list(self.feature.keys())
        self.feature_values = list(self.feature.values())
        for tag in self.tags:
            if tag == "STOP":
                continue
            else:
                self.actual_tags[tag] = self.tags[tag]
        for tag in self.actual_tags:
            bigram_feature = self.create_bigram_feature(tag)
            bigram_feature_id = self.get_feature_id(bigram_feature)
            self.bigram_feature_id[tag] = bigram_feature_id[0]
        START_feature = self.create_bigram_feature("START")
        START_feature_id = self.get_feature_id(START_feature)
        self.bigram_feature_id["START"] = START_feature_id[0]
        print("the total number of features is " + str(self.feature_length))
        print("the total number of tags is " + str(self.tags_length))
        print "the total number of self.bigram_feature_id is ", len(self.bigram_feature_id)
        #exit(0)
        #输出特征
        #zy = open("zy_feature.txt", mode = 'w')
        #for f in self.feature:
            #zy.write(f.encode('utf-8')+'\n')

    def dot(self, f_id, offset):
        score = 0
        for f in f_id:
            score += self.w[offset + f]
        return score

    def dot_v(self, f_id, offset):
        score = 0
        for f in f_id:
            score += self.v[offset + f]
        return score

    def get_feature_id(self, fv):
        fv_id = []
        for feature in fv:
            if(feature in self.feature):
                fv_id.append(self.feature[feature])
        return fv_id;

    """函数名称：Score
       函数功能：给定feature_id，求当前位置的词性为ti的总得分 0 <= i <= sentence_length
       函数返回：当前feature_id的总得分"""
    def Score(self, feature_id, ti):
        score = 0.0
        score = self.dot(feature_id, self.feature_length * self.tags[ti])
        return score

    """函数名称：Score_v
       函数功能：给定feature_id，求当前位置的词性为ti的总得分 0 <= i <= sentence_length
       函数返回：当前feature_id的总得分"""
    def Score(self, feature_id, ti):
        score = 0.0
        score = self.dot_v(feature_id, self.feature_length * self.tags[ti])
        return score

    """函数名称：Forward
       函数功能：前向Viterbi，计算句子S的所有可能的词性序列得分之和 -1 <= k <= sentence_length - 1
       函数返回：句子的得分之后"""
    def Forward(self, sentence_unigram_feature_id, k, t):
        if k == -1:
            return 1.0
        dict_current_score = {}
        final_score = 0.0
        for tag in self.actual_tags:  #初始化alpha(0, t)
            score = 0.0
            score += math.e ** self.Score(sentence_unigram_feature_id[0], tag) * 1.0
            score *= math.e ** self.Score([self.bigram_feature_id["START"]], tag)
            dict_current_score[tag] = score
        dict_left_score = dict_current_score
        if k == 0:
            return dict_left_score[t]
        for i in range(1, k):
            dict_current_score = {}
            for tag in self.actual_tags:  #对于某一个具体的tag，求alpha(k, t)
                score = 0.0
                unigram_score = math.e ** self.Score(sentence_unigram_feature_id[i], tag)
                for tag_left in self.actual_tags:
                    score += unigram_score * math.e ** self.Score([self.bigram_feature_id[tag_left]], tag) * dict_left_score[tag_left]
                dict_current_score[tag] = score
            dict_left_score = dict_current_score
        if k == len(sentence_unigram_feature_id):
            final_score = sum(list(dict_left_score.values()))
            return final_score
        #计算最后一步
        score = 0.0
        unigram_score = math.e ** self.Score(sentence_unigram_feature_id[k], t)
        for tag_left_tag in self.actual_tags:
            score += unigram_score * math.e ** self.Score([self.bigram_feature_id[tag_left_tag]], t) * dict_left_score[tag_left_tag]
        final_score = score
        return final_score

    """函数名称：Backward
       函数功能：后向Viterbi，计算句子S的所有可能的词性序列之和 -1 <= k <= sentence_length -1
       函数返回：句子的后向得分之和"""
    def Backward(self, sentence_unigram_feature_id, k, t):
        dict_right_score = {}
        dict_current_score = {}
        final_score = 0.0
        score = 0.0
        sentence_length = len(sentence_unigram_feature_id)
        if k == sentence_length:
            return 1.0
        #初始化beta(len - 1, tag)
        for tag in self.actual_tags:
            dict_current_score[tag] = math.e ** self.Score([self.bigram_feature_id[tag]], "STOP") * 1
        if k == sentence_length - 1:
            return dict_current_score[t]
        dict_right_score = dict_current_score
        for i in list(reversed(range(k + 1, sentence_length - 1))):
            dict_current_score = {}
            if i == 0:  #k == -1
                tag_left = "START"
                score = 0.0
                for tag in self.actual_tags:
                    score += math.e ** self.Score(sentence_unigram_feature_id[i], tag) * dict_right_score[tag]
                    score *= math.e ** self.Score([self.bigram_feature_id[tag_left]], tag)
                dict_current_score[tag_left] = score
                dict_right_score = dict_current_score
                continue
            for tag_left in self.actual_tags:
                score = 0.0
                for tag in self.actual_tags:
                    score = math.e ** self.Score(sentence_unigram_feature_id[i], tag) * dict_right_score[tag]
                    score *= math.e ** self.Score([self.bigram_feature_id[tag_left]], tag)
                dict_current_score[tag_left] = score
            dict_right_score = dict_current_score
        if k == -1:
            final_score = sum(dict_right_score.values())
        #计算最后一步
        score = 0.0
        for tag_right_tag in self.actual_tags:
            score = math.e ** self.Score(sentence_unigram_feature_id[k + 1], tag_right_tag) * dict_right_score[tag_right_tag]
            score += math.e ** self.Score([self.bigram_feature_id[t]], tag_right_tag)
        final_score = score
        return final_score

    """函数名称：update_gradient
       函数功能：更新梯度gradient"""
    def update_gradient(self, sentence):
        sentence_len = len(sentence.word)
        sentence_unigram_feature_id = []
        sentence_bigram_feature_id = []
        for i in range(sentence_len):  #得到整个句子的unigram特征id，sentence_unigram_feature_id 和 bigram特征id，sentence_bigram_feature_id
            if i == 0:
                ti_left_tag = "START"
            else:
                ti_left_tag = sentence.tag[i - 1]
            unigram_feature = self.create_unigram_feature(sentence, i)
            unigram_feature_id = self.get_feature_id(unigram_feature)
            bigram_feature = self.create_bigram_feature(ti_left_tag)
            bigram_feature_id = self.get_feature_id(bigram_feature)
            sentence_unigram_feature_id.append(unigram_feature_id)
            sentence_bigram_feature_id.append(bigram_feature_id)
        for i in range(sentence_len):  #g = g + f(S, Y)
            ti = sentence.tag[i]
            unigram_feature_id = sentence_unigram_feature_id[i]
            bigram_feature_id = sentence_bigram_feature_id[i]
            for index in unigram_feature_id:
                self.g[self.feature_length * self.tags[ti] + index] += 1
            for index in bigram_feature_id:
                self.g[self.feature_length * self.tags[ti] + index] += 1
        dinominator = self.Forward(sentence_unigram_feature_id, sentence_len, "STOP")  #得到分母Z(S)
        for i in range(sentence_len):  # g = g - \sum_{i = 1}^{n} p * f
            for ti in self.actual_tags:
                unigram_feature_id_i = sentence_unigram_feature_id[i]
                bigram_feature_id_i = sentence_bigram_feature_id[i]
                score = self.Score(unigram_feature_id_i, ti)
                score += self.Score(bigram_feature_id_i, ti)
                score = math.e ** score
                p = self.Forward(sentence_unigram_feature_id, i - 1, ti_left_tag) * score * self.Backward(sentence_unigram_feature_id, i, ti) / dinominator
                for index in unigram_feature_id_i:
                    self.g[self.feature_length * self.tags[ti] + index] -= p
                for index in bigram_feature_id_i:
                    self.g[self.feature_length * self.tags[ti] + index] -= p
            print "\rword ", i
            sys.stdout.flush()

    def update_weight(self):
        self.w = [x + y for x, y in zip(self.w, self.g)]

    def max_tag_sequence(self, sentence):
        list_sentence_tag = []  #用来保存路径
        tag_sequence = []
        last_score = dict()
        current_score = dict()
        last_tag = dict()
        current_tag = dict()
        maxscore = float("-Inf")
        maxtag = "NULL"
        sentence_length = len(sentence.word)
        sentence_unigram_feature_id = []
        sentence_bigram_feature_id = []
        #获取整个句子的unigram和bigram
        for i in range(sentence_length):
            if i == 0:
                ti_left_tag = "START"
            else:
                ti_left_tag = sentence.tag[i - 1]
            unigram_feature = self.create_unigram_feature(sentence, i)  #获取句子当前位置i的unigram
            unigram_feature_id = self.get_feature_id(unigram_feature)
            sentence_unigram_feature_id.append(unigram_feature_id)
            bigram_feature = self.create_bigram_feature(ti_left_tag)  #获取以ti_left_tag为左边一个词的bigram
            bigram_feature_id = self.get_feature_id(bigram_feature)
            sentence_bigram_feature_id.append(bigram_feature_id)
        #第一个词
        for t in self.tags:
            tempscore = self.Score_v(sentence_unigram_feature_id[0], t) + self.Score(sentence_bigram_feature_id[0], t)
            current_score[t] = tempscore
            current_tag[t] = "START"
        list_sentence_tag.append(current_tag)
        #如果只有一个词
        if(sentence_length == 1):
            maxscore = float("-Inf")
            maxtag = "NULL"
            for tag in current_score.keys():
                c_score = current_score[tag]
                if(c_score > maxscore):
                    maxscore = c_score
                    maxtag = tag
        ith_max_tag = maxtag
        last_score = current_score
        current_score = dict()
        last_tag = current_tag
        current_tag = dict()
        #后几个词
        for pos in range(1, sentence_length):
            maxscore = float("-Inf")
            maxtag = "NULL"
            current_score = dict()
            current_tag = dict()
            for tag in self.tags:
                maxscore = float("-Inf")
                maxtag = "NULL"
                unigram_feature = self.Score(sentence_unigram_feature_id[i], tag)
                for t in last_score.keys():
                    tempscore = unigram_score + self.Score([self.bigram_feature_id[t]], tag)
                    if(tempscore > maxscore):
                        maxscore = tempscore
                        maxtag = t
                current_score[tag] = maxscore
                current_tag[tag] = maxtag
            #找到上一个词最大的tag
            maxscore = float("-Inf")
            maxtag = "NULL"
            for tag in current_score.keys():
                if(current_score[tag] > maxscore):
                    maxscore = current_score[tag]
                    maxtag = tag
            #tag_sequence.append(current_tag[maxtag])
            last_score = current_score
            last_tag = current_tag
            ith_max_tag = maxtag
            list_sentence_tag.append(current_tag)
        tag_sequence.append(ith_max_tag)
        list_sentence_tag.reverse()
        #print list_sentence_tag[-2]
        for i in range(len(list_sentence_tag) - 1):
            ith_max_tag = list_sentence_tag[i][ith_max_tag]
            tag_sequence.append(ith_max_tag)
        tag_sequence.reverse()
        #print len(sentence.word)
        #print len(tag_sequence)
        #print tag_sequence
        #print sentence.tag
        return tag_sequence

    """函数名称：perceptron_online_training
       函数功能：对global linear model进行训练
       函数返回：无"""
    def perceptron_online_training(self):
        max_train_precision, max_dev_precision, update_times, batch, B = 0.0, 0.0, 0, 0, 1
        sentence_count = 0
        self.g = [0] * (self.feature_length * self.tags_length)
        for iterator in range(0, 20):  #进行20次迭代
            print("iterator " + str(iterator))
            for s in self.train.sentences:
                self.update_gradient(s)
                batch += 1
                sentence_count += 1
                print "\rsentence ", sentence_count
                sys.stdout.flush()
                if batch == B:  #batch is over, update weight
                    self.update_weight()
                    batch = 0
                    self.g = [0] * (self.feature_length * self.tags_length)
            #本次迭代结束，如果还有g没有更新，进行最后的更新
            if batch > 0:
                self.update_weight()
                batch = 0
                self.g = [0] * (self.feature_length * self.tags_length)
            #self.save_model(iterator)
            #进行评估
            dev_iterator, dev_c, dev_count, dev_precision = self.evaluate(self.dev, iterator)
            #保存概率最大的情况
            if(dev_precision > (max_dev_precision + 1e-10)):
                max_dev_precision, max_dev_iterator, max_dev_c, max_dev_count = dev_precision, dev_iterator, dev_c, dev_count
        print("Conclusion:")
        print("\t"+self.dev.name + " iterator: "+str(max_dev_iterator)+"\t"+str(max_dev_c)+" / "+str(max_dev_count) + " = " +str(max_dev_precision))

    def save_model(self, iterator):
        fmodel = open("linearmodel.lm"+str(iterator), mode='w')
        for feature_id in self.feature_values:
            feature = self.feature_keys[feature_id]
            left_feature = feature.split(':')[0] + ':'
            right_feature = '*' + feature.split(':')[1]
            for tag in self.actual_tags:
                tag_id = self.tags[tag]
                entire_feature = left_feature + tag + right_feature
                w = self.w[tag_id * self.feature_length + feature_id]
                if(w != 0):
                    fmodel.write(entire_feature.encode('utf-8') + '\t' + str(w) + '\n')
        fmodel.close()

    """函数名称：evaluate
       函数功能：根据开发集dev，测试global linear model训练得到的成果
                 输出正确率
       函数返回：迭代次数，正确的tag的个数，所有的word的个数，准确率"""
    def evaluate(self, dataset, iterator):
       c = 0  #记录标注正确的tag数量
       for s in dataset.sentences:
           max_tag_sequence = self.max_tag_sequence(s)  #使用v进行评估，返回得分最高的tag序列
           correct_tag_sequence = s.tag
           for i in range(len(max_tag_sequence)):  #比较每一个tag，是否相等？
               if(max_tag_sequence[i] == correct_tag_sequence[i]):
                   c += 1
       accuracy = 1.0 * c / dataset.total_word_count
       print(dataset.name + "\tprecision is " + str(c) + " / " + str(dataset.total_word_count) + " = " + str(accuracy))
       return iterator, c, dataset.total_word_count, accuracy


################################ main #####################################
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    crf = conditional_random_field()
    crf.create_feature_space()  #创建特征空间
    crf.perceptron_online_training()  #global linear model perceptron online training
    endtime = datetime.datetime.now()
    print("executing time is "+str((endtime-starttime).seconds)+" s")
