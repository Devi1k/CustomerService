import numbers
import os.path
import time

# import jieba.analyse
import numpy as np
import pkuseg

from ..utils.ai_wrapper import get_related_title

setattr(time, "clock", time.perf_counter)

_similarity_smooth = lambda x, y, z, u: (x * y) + z - u
pku = pkuseg.pkuseg(
    user_dict=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/new_dict.txt'))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def is_digit(obj):
    '''
    Check if an object is Number
    '''
    return isinstance(obj, (numbers.Integral, numbers.Complex, numbers.Real))


def load_dict(file_path):
    word_dictionary = []
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        for word in content:
            word_dictionary.append(word.strip())
    return word_dictionary


def lev(first, second, utterance=False, service=False):
    """

    :param first: utterance / candidate service
    :param second: service name
    :param utterance:是否计算utterance与事项
    :param service:是否诊断前的阈值计算
    :return:
    """
    sentence1_len, sentence2_len = len(first), len(second)
    maxlen = max(sentence1_len, sentence2_len)
    if not service:
        if sentence1_len > sentence2_len:
            first, second = second, first

    distances = range(len(first) + 1)
    count = 0
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                              distances[index1 + 1],
                                              new_distances[-1])))
        distances = new_distances
    if utterance:
        for c2 in second:
            if c2 in first: count += 1

    levenshtein = distances[-1]
    d = float((maxlen - levenshtein) / maxlen)
    s = d + count / len(second)
    # smoothing
    if not utterance and not service:
        s = (sigmoid(s * 6) - 0.5) * 2
    # print("smoothing[%s| %s]: %s -> %s" % (sentence1, sentence2, d, s))
    return s


def compare(s1, s2, model):
    g = 0
    try:
        g_ = model.wv.similarity(s1, s2)
        if is_digit(g_): g = g_
    except:
        pass
    u = lev(s1, s2)
    if u >= 0.99:
        r = 1.0
    elif u > 0.9:
        r = _similarity_smooth(g, 0.05, u, 0.05)
    elif u > 0.8:
        r = _similarity_smooth(g, 0.1, u, 0.2)
    elif u > 0.4:
        r = _similarity_smooth(g, 0.2, u, 0.15)
    elif u > 0.2:
        r = _similarity_smooth(g, 0.3, u, 0.1)
    else:
        r = _similarity_smooth(g, 0.4, u, 0)

    if r < 0: r = abs(r)
    r = min(r, 1.0)
    return float("%.3f" % r)


def replace_list(seg_list, word_dict, similarity_dict, model):
    new_list = set()
    for x in seg_list:
        replace_word = x
        max_score = 0
        to_check = []
        u = x
        # seek_start = time.time()
        try:
            u = similarity_dict[x]
            # u = model.wv.most_similar(x, topn=5)
        except KeyError:
            pass
        to_check.append(x)
        # seek_end = time.time()
        # print("seek:", seek_end - seek_start)
        for _u in u:
            to_check.append(_u)
        to_check = list(reversed(to_check))
        # com_start = time.time()
        for k in to_check:
            score = [compare(k, y, model) for y in word_dict]
            choice = max(score)
            if choice >= max_score:
                max_score = choice
                choice_index = int(score.index(choice))
                replace_word = list(word_dict)[choice_index]
                # if check_score > 0.1:
                #     replace_word = check_word
        # com_end = time.time()
        # print("compare:", com_end - com_start)
        new_list.add(replace_word)
    return list(new_list)


def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]





def cut_sentence_remove_stopwords(sentence):
    stop_words = [i.strip() for i in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       'data/baidu_stopwords.txt')).readlines()]
    seg = pku.cut(sentence)
    seg_list = []
    for s in seg:
        seg_list.append(s)
    for i in range(len(seg_list) - 1, -1, -1):
        if seg_list[i] in stop_words:
            del seg_list[i]
    return seg_list
