import json
import os.path
import re
import signal
import warnings
from json import JSONDecodeError

import Levenshtein
import requests

from ..utils.message_sender import messageSender

warnings.filterwarnings("ignore")

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/faq_recommend.json'),
          'r') as f:
    recommend = json.load(f)

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/faq_with_service.json'),
          'r') as f:
    new_recommend = json.load(f)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/syn_service.json'),
          'r') as f:
    syn_service = json.load(f)


def get_item_content(first_utterance):
    item_content = "https://miner.picp.net/chatBot/matterContent?text={}"
    try:
        res = requests.get(item_content.format(first_utterance), verify=False).json()['data']
    except Exception:
        res = requests.get(item_content.format(first_utterance), verify=False).json()['data']
    if res == "I":
        return "item"
    else:
        return "content"


# 常见问题回答
def get_faq(first_utterance, service=""):
    faq_path = "https://miner.picp.net/FAQ?First_utterance={}"
    first_utterance = service + first_utterance
    try:
        faq_res = requests.get(faq_path.format(first_utterance), verify=False).json()
        similar_score, answer, service = faq_res['Similarity_score'], faq_res['answer'], faq_res['service']
    except JSONDecodeError:
        similar_score, answer, service = 1, "抱歉，网络错误，请您重新尝试", ""
    return similar_score, answer, service


# 业务
def get_business(first_utterance):
    business_path = "https://miner.picp.net/yewu?text={}"
    res = requests.get(business_path.format(first_utterance), verify=False).json()
    return res['type']


# 文档检索
def get_retrieval(first_utterance, service_name):
    ir_path = "https://burninghell.xicp.net/IR?serviceName={}&firstUtterance={}"
    ir_res = requests.get(ir_path.format(service_name, first_utterance), verify=False).json()['abs']
    return ir_res


# 业务推理
def get_nli(first_utterance, service_name):
    nli_path = "https://burninghell.xicp.net/zmytest?Service_name={}&First_utterance={}"
    nli_res = requests.get(nli_path.format(service_name, first_utterance), verify=False).text
    return nli_res


# 进入对话后的检索事项
def get_related_title(first_utterance):
    from ..utils.word_match import cut_sentence_remove_stopwords
    first_utterance = ''.join(cut_sentence_remove_stopwords(first_utterance))
    title_path = "https://burninghell.xicp.net/getRelatedTitle/ver2?query={}"
    # title_res = []
    try:
        title_res = requests.get(title_path.format(first_utterance), verify=False).json()['titleList'][:5]
        if len(title_res) > 0:
            title_res.append('以上都不是')
    except:
        title_res = []

    return title_res


def get_answer(first_utterance, service_name, log, intent_class=''):
    try:
        # --intention detection
        if intent_class == '':
            intent_path = "https://miner.picp.net/chatBot/intent?text={}"
            intent_res = requests.get(intent_path.format(first_utterance), verify=False).json()
            intent_class = intent_res['data']
            if '认定高中教师资格的学历要求' in first_utterance:
                return '研究生或者大学本科学历'
            elif '16岁以上护照有效期多长' in first_utterance:
                return '十年'
            log.info("intention:{}".format(intent_class))

        if intent_class == "QA":  # --QA match
            answer = get_retrieval(first_utterance, service_name)
            log.info("QA: {}".format(answer))
            return answer

        # 业务推理
        elif intent_class == "NLI":  # --NLI
            nli_res = get_nli(first_utterance, service_name)
            log.info("NLI:{} ".format(nli_res))
            return nli_res

        # 文档检索
        elif intent_class == "IR":  # --IR
            ir_res = get_retrieval(first_utterance, service_name)
            log.info("IR: {}".format(ir_res))
            return ir_res

    except Exception:
        pass


def faq_diagnose(answer, dialogue_content, conv_id, log, service_name=""):
    # 子进程管道关闭
    # user_pipe[0].close()
    # response_pipe[1].close()
    # user_pipe[1].close()
    # response_pipe[0].close()

    # 对话状态设置
    dialogue_content[4] = True
    dialogue_content[6] = True

    messageSender(conv_id=conv_id, msg=answer, log=log, end=dialogue_content[4])

    dialogue_content[2] = ""
    # dialogue_content[3].kill()
    if dialogue_content[3] != 0:
        os.kill(dialogue_content[3], signal.SIGKILL)
    # log.info('process kill')
    # last_msg = "请问还有其他问题吗，如果有请继续提问"
    # messageSender(conv_id=conv_id, msg="请问还有其他问题吗，如果有请继续提问", log=log, end=dialogue_content[4])

    # FAQ推荐 后续实现
    recommend = get_recommend(service_name=dialogue_content[7],
                              history=dialogue_content[10])
    if len(recommend) < 1:
        recommend = "请问还有其他问题吗，如果有请继续提问"
    last_msg = recommend
    if isinstance(recommend, list):
        messageSender(conv_id=conv_id, msg="大家都在问", log=log, end=True,
                      service_name=service_name, options=last_msg)
    else:
        messageSender(conv_id=conv_id, msg=last_msg, log=log, end=dialogue_content[4])
    dialogue_content[2] = ""
    dialogue_content[9] = 0
    dialogue_content[8] = last_msg
    return dialogue_content


def get_faq_from_service(first_utterance, service, history):
    from .word_match import cut_sentence_remove_stopwords
    syn_list = []
    if service in syn_service.keys():
        syn_list = syn_service[service]
        for s in syn_list:
            if s in first_utterance:
                first_utterance = first_utterance.replace(s, "")
        first_utterance = first_utterance.replace(service, "")
    utterance = first_utterance.replace("--", '-').replace(" ", "")
    seg_list = cut_sentence_remove_stopwords(sentence=utterance)
    utterance = ''.join(seg_list)
    utterance = re.sub("[\s++\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*]+", "",
                       utterance)
    try:
        question_dict = new_recommend[service]
    except KeyError:
        return 0, "抱歉，无相关问题", service
    question_list = set()
    for k, v in question_dict.items():
        for ques, document in v.items():
            if len(syn_list) > 0:
                for s in syn_list:
                    if s in ques:
                        question_list.add(ques.replace(s, ""))
            question_list.add(ques.replace(service, ""))
    question_list = list(question_list)
    answer = ""
    max_score = 0
    candidate_ques = ""
    # first_utterance = first_utterance.replace(service, "")
    max_num = 0
    for q in question_list:
        _q = re.sub("[\s++\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*]+", "",
                    q)
        num = 0
        for s in seg_list:
            if s in _q:
                num += 1
        scoreT = Levenshtein.ratio(utterance, _q)
        if num >= max_num and scoreT > max_score:
            max_score = scoreT
            max_num = num
            candidate_ques = q
    for k, v in question_dict.items():
        if candidate_ques in v.keys() or (service + candidate_ques) in v.keys():
            try:
                answer = v[candidate_ques]
            except KeyError:
                answer = v[service + candidate_ques]
            break
    return max_score, answer, service


def return_answer(dialogue_content, conv_id, service_name, log, link, item_content="content", intent_class=''):
    if item_content == "content":
        similarity_score, answer, faq_service = get_faq_from_service(first_utterance=dialogue_content[2],
                                                                     service=service_name, history=dialogue_content[10])
        if float(similarity_score) < 0.34:
            answer = get_answer(first_utterance=dialogue_content[2], service_name=dialogue_content[7], log=log,
                                intent_class=intent_class)
    else:
        answer = "您询问的业务属于:" + service_name
    try:
        service_link = str(link[service_name])
    except KeyError:
        service_link = ""
    business = get_business(first_utterance=dialogue_content[2])
    answer = answer + '\n' + '(' + service_name + '——' + business + ')'
    dialogue_content[10].append(dialogue_content[2])
    messageSender(conv_id=conv_id, msg=answer, log=log, link=service_link, end=True)
    dialogue_content[4] = True
    dialogue_content[6] = True
    # pipes_dict[conv_id][3].kill()
    if dialogue_content[3] != 0:
        os.kill(dialogue_content[3], signal.SIGKILL)
        log.info('process kill')
    recommend = get_recommend(service_name=dialogue_content[7],
                              history=dialogue_content[10])
    if len(recommend) < 1:
        recommend = "请问还有其他问题吗，如果有请继续提问"
    dialogue_content[8] = recommend
    log.info('provide recommend')
    if isinstance(recommend, list):
        messageSender(conv_id=conv_id, msg="大家都在问", log=log, end=True,
                      service_name=service_name, options=dialogue_content[8])
    else:
        messageSender(conv_id=conv_id, msg=dialogue_content[8], log=log, end=dialogue_content[4])
    dialogue_content[2] = ""
    dialogue_content[9] = 0
    return dialogue_content


def get_multi_res(first_utterance, service_name):
    answer = get_retrieval(first_utterance=first_utterance, service_name=service_name)
    business = get_business(first_utterance=first_utterance)
    answer = answer + '\n' + '(' + service_name + '——' + business + ')'
    return answer


def get_recommend(service_name, history=None):
    if history is None:
        history = []
    level_list = ['1', '2', '3']
    query_list = []
    for level in level_list:
        try:
            query = recommend[service_name][level]
        except KeyError:
            continue
        for q in query:
            q = q.replace(service_name, "")
            q = re.sub("[\s++\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*]+",
                       "",
                       q)
            query_list.append(q)
    if history is not None:
        for h in history:
            for i in range(len(query_list) - 1, -1, -1):
                q = query_list[i]
                scoreT = Levenshtein.ratio(h, q)
                if scoreT > 0.38:
                    query_list.remove(q)
    return query_list[:5]
