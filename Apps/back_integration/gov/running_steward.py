# -*-coding: utf-8 -*-
import json
import os
import signal

from Apps.back_integration.gov.dialogue_manager import DialogueManager
from Apps.back_integration.gov.user import User


def simulation_epoch(pipe, agent, parameter, log, similarity_dict, conv_id, train_mode=1):
    in_pipe, out_pipe = pipe
    user = User(parameter=parameter, conv_id=conv_id)
    agent = agent
    dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter, log=log,
                                       similarity_dict=similarity_dict)
    dialogue_manager.set_agent(agent=agent)
    positive_list = ['是的', '是', '没错', '对', '对的,', '嗯']
    episode_over = False
    explicit = ""
    try:
        receive = in_pipe.recv()
        explicit = receive
    except EOFError:
        episode_over = True
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)
    # init_start = time.time()

    agent_action = dialogue_manager.initialize(explicit, train_mode=parameter.get("train_mode"),
                                               greedy_strategy=0)
    # init_end = time.time()
    # print("init:", init_end - init_start)

    if agent_action['action'] == 'inform':
        msg = {"service": agent_action["inform_slots"]["service"],
               "action": agent_action['action'],
               "end_flag": episode_over}
        out_pipe.send(msg)
    elif agent_action['action'] == 'request':
        send_list = list(agent_action["request_slots"].keys())
        service = ''.join(send_list)
        msg = {"service": service,
               "action": agent_action['action'],
               "end_flag": episode_over}
        log.info(msg)
        out_pipe.send(msg)

    while True:
        if episode_over is True:
            if agent_action['action'] == 'inform':
                msg = {"service": agent_action["inform_slots"]["service"],
                       "action": agent_action['action'],
                       "end_flag": episode_over}
                out_pipe.send(msg)
            elif agent_action['action'] == 'request':
                msg = {"service": None,
                       "action": agent_action['action'],
                       "end_flag": episode_over}
                log.info(msg)
                out_pipe.send(msg)
            break
        try:
            receive = in_pipe.recv()
        except EOFError:
            break
        # judge = receive['judge']
        judge = False
        implicit = receive
        # print(implicit)
        if implicit in positive_list:

            judge = True
            with open(user.goal_set_path, 'r') as f:
                goal_set = json.load(f)
                goal_set['user_action']['user_judge'] = True
            with open(user.goal_set_path, 'w') as f:
                json.dump(goal_set, f)
            implicit = ""
        else:
            with open(user.goal_set_path, 'r') as f:
                goal_set = json.load(f)
                goal_set['user_action']['user_judge'] = False
            with open(user.goal_set_path, 'w') as f:
                json.dump(goal_set, f)
                # episode_over = True
                # break
        if agent_action['action'] == 'inform' and judge is True:
            # out_pipe.send("请问还有别的问题吗")
            episode_over = True
            msg = {"service": agent_action["inform_slots"]["service"],
                   "action": agent_action['action'],
                   "end_flag": episode_over}
            out_pipe.send(msg)
            break
        # next_start = time.time()
        reward, episode_over, dialogue_status, _agent_action = dialogue_manager.next(implicit,
                                                                                     save_record=True,
                                                                                     train_mode=train_mode,
                                                                                     greedy_strategy=0,
                                                                                     agent_action=agent_action)
        # next_end = time.time()
        # print("next:", next_end - next_start)
        agent_action = _agent_action
        log.info(agent_action)
        if agent_action['action'] == 'inform':
            msg = {"service": agent_action["inform_slots"]["service"],
                   "action": agent_action['action'],
                   "end_flag": episode_over}
            out_pipe.send(msg)
        elif agent_action['action'] == 'request':
            send_list = list(agent_action["request_slots"].keys())
            service = ''.join(send_list)
            msg = {"service": service,
                   "action": agent_action['action'],
                   "end_flag": episode_over}
            out_pipe.send(msg)

    in_pipe.close()
    out_pipe.close()
