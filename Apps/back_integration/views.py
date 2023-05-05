import os
from multiprocessing import Pipe, Process

import gensim
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .conf.config import get_config
from .gov.agent_rule import AgentRule
from .gov.running_steward import simulation_epoch
from .utils.ResultGenerator import ResultGenerator
from .utils.ai_wrapper import *
from .utils.logger import Logger
from .utils.message_sender import messageSender
from .utils.word_match import *

log = Logger('integration').getLogger()
warnings.filterwarnings("ignore")

parent_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(parent_path, 'data/blur_service.json'), 'r') as f:
    blur_service = json.load(f)
model = gensim.models.Word2Vec.load(os.path.join(parent_path, 'data/wb.text.model'))
word_dict = load_dict(os.path.join(parent_path, 'data/new_dict.txt'))
stop_words = [i.strip() for i in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'data/baidu_stopwords.txt')).readlines()]
positive_list = ['是的', '是', '没错', '对', '对的,', '嗯', 'sheji']
hello_list = ["你好", "您好", "打扰了"]
link_file = os.path.join(parent_path, 'data/link.json')
with open(link_file, 'r') as f:
    link = json.load(f)
with open(os.path.join(parent_path, 'data/similar.json'), 'r') as f:
    similarity_dict = json.load(f)
config_file = os.path.join(parent_path, 'conf/settings.yaml')

parameter = get_config(config_file)
# agent = AgentDQN(parameter=parameter)

agent = AgentRule(parameter=parameter)
pipes_dict = {}


@csrf_exempt
def chat(request):
    result = ResultGenerator()
    if request.method == "GET":
        response = request.body.decode("utf-8")
        # log.info(response)
        try:
            response = json.loads(response)
        except JSONDecodeError:
            return JsonResponse(result.getFailResult("Json Format Error"), safe=False)
        process_msg(response)
        return JsonResponse(result.getNothingResult(), safe=False)
    else:
        return JsonResponse(result.getFailResult("Request Method Error"), safe=False)


def process_msg(user_json):
    global pipes_dict, agent, parameter, link, similarity_dict, positive_list, stop_words, word_dict, model, \
        blur_service
    # result = ResultGenerator()
    # log.info('exec process_msg.child process id : %s, parent process id : %s' % (os.getpid(), os.getppid()))

    msg = user_json['msg']
    conv_id = msg['conv_id']
    if 'content' in msg.keys():
        log.info("user message:" + msg['content']['text']) if 'text' in msg['content'].keys() else log.info(
            "user choice:" + msg['content']['service_name'])
    if "content" in msg.keys() and "text" in msg["content"].keys() and msg["content"]["text"] in hello_list:
        messageSender(conv_id=conv_id, log=log,
                      msg="欢迎您使用北辰客服智能，我是您的智能助手小辰，负责事项、业务以及相关问题的咨询。有什么问题您尽管说。")
        return
    try:
        if msg['content']['service_name'] is not None:
            start_time = time.time()
            service_name = msg['content']['service_name']
            dialogue_content = pipes_dict[conv_id]
            if service_name != "以上都不是" and dialogue_content[9] != 0:
                dialogue_content[7] = service_name.replace("--", "-")
                item_content = get_item_content(dialogue_content[2])
                if dialogue_content[2] != "":
                    log.info("add message to history")
                    dialogue_content[10].append(dialogue_content[2])
                log.info("item content:{}".format(item_content))
                dialogue_content = return_answer(dialogue_content=dialogue_content, conv_id=conv_id,
                                                 service_name=dialogue_content[7],
                                                 log=log,
                                                 link=link, item_content=item_content)
                pipes_dict[conv_id] = dialogue_content
                log.info("user choose service and return answer cost: {}".format(str(time.time() - start_time)))
                return
            elif service_name == '以上都不是':
                if dialogue_content[9] > 1:
                    dialogue_content[4] = True
                    dialogue_content[8] = "抱歉，未能找到您所需的事项。请问还有其他问题吗，如果有请继续提问。"
                    dialogue_content[7] = ""
                    dialogue_content[10] = []
                    dialogue_content[9] = 0
                    messageSender(conv_id=conv_id, msg=dialogue_content[8], log=log,
                                  end=dialogue_content[4])
                    dialogue_content[6] = True
                    dialogue_content[2] = ""
                    # pipes_dict[conv_id][3].kill()
                    if dialogue_content[3] != 0:
                        os.kill(dialogue_content[3], signal.SIGKILL)
                    log.info('process kill')
                    pipes_dict[conv_id] = dialogue_content
                    return
                elif dialogue_content[9] == 1:
                    messageSender(conv_id=conv_id, msg="抱歉，我没太理解您的意思。请再补充一些细节呢。", log=log,
                                  end=False)
                    return
            elif service_name != '以上都不是' and dialogue_content[9] == 0:
                start_time = time.time()
                dialogue_content[2] = service_name.replace("--", "-")
                dialogue_content[10].append(dialogue_content[2])
                dialogue_content[4] = True
                dialogue_content[6] = True

                similarity_score, answer, service_name = get_faq_from_service(
                    first_utterance=dialogue_content[2],
                    service=dialogue_content[7], log=log)
                dialogue_content[7] = service_name
                messageSender(conv_id=conv_id, msg=answer, log=log, end=dialogue_content[4])
                recommend = get_recommend(service_name=dialogue_content[7],
                                          history=dialogue_content[10])
                if len(recommend) < 1:
                    recommend = "请问还有其他问题吗，如果有请继续提问"
                dialogue_content[8] = recommend
                if isinstance(recommend, list):
                    messageSender(conv_id=conv_id, msg="大家都在问", log=log, end=True,
                                  service_name=service_name, options=recommend)
                else:
                    messageSender(conv_id=conv_id, msg=recommend, log=log, end=dialogue_content[4])
                dialogue_content[2] = ""
                dialogue_content[9] = 0
                pipes_dict[conv_id] = dialogue_content
                log.info("user choose faq and return answer cost: {}".format(str(time.time() - start_time)))

                return
    except KeyError:
        pass
    try:
        # Initialize the conversation
        if conv_id not in pipes_dict:
            user_pipe, response_pipe = Pipe(), Pipe()
            log.info("new conv")
            pipes_dict[conv_id] = [user_pipe, response_pipe, "", 0, False, False, True,
                                   "", "", 0, []]
        # Handle multiple rounds of dialogues  Continue to speak
        elif conv_id in pipes_dict and pipes_dict[conv_id][5] is False and pipes_dict[conv_id][4] is True:
            start_time = time.time()
            log.info("continue to ask")
            dialogue_content = pipes_dict[conv_id]
            # todo: wait to update new front end, delete this
            if 'content' not in msg.keys():
                if isinstance(dialogue_content[8], list):
                    messageSender(conv_id=conv_id, options=dialogue_content[8], log=log)
                else:
                    messageSender(conv_id=conv_id, msg=dialogue_content[8], log=log)
                return
            multi = True
            if dialogue_content[2] == "":
                try:
                    dialogue_content[2] = msg['content']['text'].replace("--", "-")
                except KeyError:
                    dialogue_content[2] = msg['content']['service_name'].replace("--", "-")
                dialogue_content[2] = re.sub("[\s++\.\!\/_$%^*]+|[+——()?【】“”、~@#￥%……&*]+|[0-9]+",
                                             "",
                                             dialogue_content[2].replace("--", "-"))
            try:
                prev_msg = ""
                for msg in dialogue_content[10]:
                    if msg != "":
                        prev_msg = msg
                log.info("cur_msg:{},pre_msg:{}".format(dialogue_content[2], prev_msg))
                if dialogue_content[10][-1] == '' or len(dialogue_content[10]) == 0:
                    multi = False
                else:
                    multi = is_multi_round(pre_text=prev_msg, cur_text=dialogue_content[2])
            except IndexError:
                multi = False
            if multi:
                log.info("Same matter.")
                dialogue_content[4] = True
                dialogue_content[6] = True
                if dialogue_content[7] not in blur_service.keys():
                    item_content = get_item_content(dialogue_content[2])
                    dialogue_content[10].append(dialogue_content[2])
                    log.info("item content:{}".format(item_content))
                    dialogue_content = return_answer(dialogue_content=dialogue_content, conv_id=conv_id,
                                                     service_name=dialogue_content[7],
                                                     log=log,
                                                     link=link, item_content=item_content)
                    pipes_dict[conv_id] = dialogue_content

                    log.info("same service and return answer cost: {}".format(str(time.time() - start_time)))

                else:
                    similarity_score, answer, service_name = get_faq_from_service(first_utterance=dialogue_content[2],
                                                                                  service=dialogue_content[7],
                                                                                  log=log)
                    log.info(str(round(similarity_score, 2)) + "  " + answer)
                    if similarity_score > 0.285:
                        # messageSender(conv_id=conv_id, msg=answer, log=log, end=True)
                        dialogue_content[7] = service_name
                        dialogue_content[10].append(dialogue_content[2])
                        dialogue_content = faq_diagnose(answer, dialogue_content, conv_id,
                                                        log, service_name=service_name)
                        pipes_dict[conv_id] = dialogue_content
                        log.info(
                            "multi round blur service and return faq answer cost: {}".format(
                                str(time.time() - start_time)))
                    else:
                        options = get_related_title(dialogue_content[7])
                        dialogue_content[9] += 2
                        messageSender(conv_id=conv_id, msg="请选择与您相符的事项", log=log, options=options,
                                      end=False)
                        log.info("multi round same service with blur service and return answer cost: {}".format(
                            str(time.time() - start_time)))

            else:
                log.info("Different matter")
                # Rediagnosis
                user_pipe, response_pipe = Pipe(), Pipe()
                p = Process(target=simulation_epoch,
                            args=(
                                (user_pipe[1], response_pipe[0]), agent, parameter, log, similarity_dict,
                                conv_id))
                dialogue_content[0], dialogue_content[1] = user_pipe, response_pipe
                dialogue_content[2] = re.sub("[\s++\.\!\/_$%^*]+|[+——()?【】“”、~@#￥%……&*]+|[0-9]+",
                                             "",
                                             dialogue_content[2].replace("--", "-"))
                user_text = {'text': dialogue_content[2]}
                log.info(user_text)
                similar_score, answer = 0, ""
                item_content = get_item_content(dialogue_content[2])
                log.info("item content:{}".format(item_content))
                if item_content == "content":
                    similar_score, answer, service_name = get_faq(dialogue_content[2])
                    if similar_score > 0.955:
                        dialogue_content[7] = service_name
                        dialogue_content[10].append(dialogue_content[2])
                        dialogue_content = faq_diagnose(answer, dialogue_content, conv_id,
                                                        log, service_name=service_name)
                        pipes_dict[conv_id] = dialogue_content
                        log.info(
                            "multi round different service and return faq answer cost: {}".format(
                                str(time.time() - start_time)))

                        return
                # After initializing the session, send judgments and descriptions to the model (including
                # subsequent judgments and supplementary descriptions).
                options = get_related_title(dialogue_content[2])
                business_threshold = 0.97
                candidate_service = ""
                max_score = 0
                for o in options:
                    lcs = longestCommonSubsequence(dialogue_content[2], o)
                    if lcs <= 2:
                        continue
                    distance = lev(dialogue_content[2], o, True, True)
                    final_distance = sigmoid(lcs / len(o) + distance)
                    if max_score < final_distance:
                        max_score = final_distance
                        candidate_service = o
                # if utterance is similar to candidate service
                if max_score > business_threshold:
                    dialogue_content[4] = True
                    dialogue_content[6] = True
                    dialogue_content[7] = candidate_service
                    log.info("first_utterance: {}".format(dialogue_content[2]))
                    log.info("service_name: {}".format(candidate_service))
                    dialogue_content[10].append(dialogue_content[2])
                    dialogue_content = return_answer(dialogue_content=dialogue_content, conv_id=conv_id,
                                                     service_name=candidate_service,
                                                     log=log,
                                                     link=link, item_content=item_content)
                    pipes_dict[conv_id] = dialogue_content
                    log.info(
                        "multi round different service with high similarity service and return answer cost: {}".format(
                            str(time.time() - start_time)))

                else:
                    dialogue_content[6] = False
                    dialogue_content[9] += 1
                    if len(options) > 0:
                        dialogue_content[4] = False
                        dialogue_content[8] = options
                        messageSender(conv_id=conv_id, log=log, options=options, end=False)
                        pipes_dict[conv_id] = dialogue_content
                        log.info("multi round different service and return related service cost: {}".format(
                            str(time.time() - start_time)))

                    else:
                        dialogue_content[4] = True
                        answer = "抱歉，未能找到您所需的事项。请问还有其他问题吗，如果有请继续提问。"
                        dialogue_content[6] = True
                        dialogue_content[10] = []
                        # if dialogue_content[3] != 0:
                        #     os.kill(dialogue_content[3], signal.SIGKILL)
                        dialogue_content[2] = ""
                        dialogue_content[9] = 0

                        messageSender(conv_id=conv_id, log=log, msg=answer, end=False)
                        dialogue_content[8] = "请问还有其他问题吗，如果有请继续提问"
                        messageSender(conv_id=conv_id, msg="请问还有其他问题吗，如果有请继续提问", log=log,
                                      end=True)
                        pipes_dict[conv_id] = dialogue_content
                        log.info("different service and diagnose failed with options cost: {}".format(
                            str(time.time() - start_time)))

        #
        else:
            start_time = time.time()
            dialogue_content = pipes_dict[conv_id]
            user_pipe, response_pipe = dialogue_content[0], dialogue_content[1]
            if 'content' not in msg.keys():
                if isinstance(dialogue_content[8], list):
                    messageSender(conv_id=conv_id, options=dialogue_content[8], log=log)
                else:
                    messageSender(conv_id=conv_id, msg=dialogue_content[8], log=log)
                return
            if dialogue_content[2] == "":
                try:
                    dialogue_content[2] = msg['content']['text'].replace("--", "-")
                except KeyError:
                    dialogue_content[2] = msg['content']['service_name'].replace("--", "-")
                dialogue_content[2] = re.sub("[\s++\.\!\/_$%^*]+|[+——()?【】“”、~@#￥%……&*]+|[0-9]+",
                                             "",
                                             dialogue_content[2].replace("--", "-"))
            else:
                if dialogue_content[9] == 1:
                    try:
                        dialogue_content[2] += msg['content']['text'].replace("--", "-")
                    except KeyError:
                        dialogue_content[2] += msg['content']['service_name'].replace("--", "-")
            # item content differ, if content get faq else judge multi round
            item_content = get_item_content(dialogue_content[2])
            log.info("item content:{}".format(item_content))
            if dialogue_content[6] is True:
                if item_content == "content":
                    similar_score, answer, service_name = get_faq(dialogue_content[2])

                    if similar_score > 0.955:
                        dialogue_content[7] = service_name
                        dialogue_content[10].append(dialogue_content[2])
                        dialogue_content = faq_diagnose(answer, dialogue_content, conv_id,
                                                        log, service_name=service_name)
                        pipes_dict[conv_id] = dialogue_content
                        log.info(
                            "different service and return faq answer cost: {}".format(str(time.time() - start_time)))

                        return
            user_text = msg['content']
            options = []
            if 'text' in user_text.keys() and dialogue_content[9] != 1:
                # IR
                options = get_related_title(dialogue_content[2])
                # 若对话内容包含的事项足够明确
                business_threshold = 0.97
                candidate_service = ""
                max_score = 0
                for o in options:
                    lcs = longestCommonSubsequence(dialogue_content[2], o)
                    if lcs <= 2:
                        continue
                    distance = lev(dialogue_content[2], o, True, True)
                    distance = sigmoid(distance + lcs / len(o))
                    if max_score < distance:
                        max_score = distance
                        candidate_service = o
                if max_score > business_threshold:
                    dialogue_content[4] = True
                    dialogue_content[6] = True
                    dialogue_content[7] = candidate_service
                    log.info("first_utterance: {}".format(dialogue_content[2]))
                    log.info("service_name: {}".format(candidate_service))
                    dialogue_content[10].append(dialogue_content[2])
                    dialogue_content = return_answer(dialogue_content=dialogue_content, conv_id=conv_id,
                                                     service_name=candidate_service,
                                                     log=log,
                                                     link=link, item_content=item_content)
                    pipes_dict[conv_id] = dialogue_content
                    log.info("different service with high similarity service and return answer cost: {}".format(
                        str(time.time() - start_time)))
                    return
            if dialogue_content[6] and len(options) > 0:
                dialogue_content[6] = False
                dialogue_content[9] += 1
                dialogue_content[8] = options
                dialogue_content[4] = False
                pipes_dict[conv_id] = dialogue_content
                messageSender(conv_id=conv_id, log=log, options=options, end=False)
                log.info("multi round different service and return related service cost: {}".format(
                    str(time.time() - start_time)))
            else:
                if dialogue_content[6]:
                    dialogue_content[6] = False
                    dialogue_content[9] += 1
                if 'text' not in user_text.keys():
                    log.info(dialogue_content[2])
                    user_text = {'text': dialogue_content[2]}
                options = get_related_title(dialogue_content[2])
                dialogue_content[9] += 1
                if len(options) > 0:
                    pipes_dict[conv_id] = dialogue_content
                    messageSender(conv_id=conv_id, log=log, options=options, end=False)
                else:
                    dialogue_content[4] = True
                    answer = "抱歉，未能找到您所需的事项。请问还有其他问题吗，如果有请继续提问。"
                    dialogue_content[6] = True
                    dialogue_content[10] = []
                    dialogue_content[2] = ""
                    dialogue_content[9] = 0

                    messageSender(conv_id=conv_id, log=log, msg=answer, end=False)
                    dialogue_content[8] = "请问还有其他问题吗，如果有请继续提问"
                    messageSender(conv_id=conv_id, msg="请问还有其他问题吗，如果有请继续提问", log=log,
                                  end=True)
                    pipes_dict[conv_id] = dialogue_content
                log.info("different service and diagnose failed with options cost: {}".format(
                    str(time.time() - start_time)))


    except Exception as e:
        log.error(e, exc_info=True)
