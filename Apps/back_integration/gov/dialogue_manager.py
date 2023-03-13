# -*- coding:utf-8 -*-
import json

import Apps.back_integration.gov.dialogue_configuration as dialogue_configuration
from Apps.back_integration.gov.state_tracker import StateTracker


class DialogueManager(object):
    def __init__(self, user, agent, parameter, log, similarity_dict):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.inform_wrong_service_count = 0
        self.similarity_dict = similarity_dict
        self.log = log

    def initialize(self, explicit_inform_slots, greedy_strategy, train_mode=1, epoch_index=None):
        self.state_tracker.initialize()
        self.inform_wrong_service_count = 0

        self.log.info(explicit_inform_slots)
        user_action = self.state_tracker.user.initialize(explicit_inform_slots)
        self.state_tracker.state_updater(user_action=user_action)

        self.state_tracker.agent.initialize()
        state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy)
        self.state_tracker.state_updater(agent_action=agent_action)
        # state = self.state_tracker.get_state()
        # self.self.log.info(state["current_slots"]["agent_request_slots"].keys())  #测试是否为空，证明不是
        return agent_action

    def set_agent(self, agent):
        self.state_tracker.set_agent(agent=agent)

    def next(self, implicit_inform_slots, save_record, train_mode, agent_action, greedy_strategy):
        # implicit_inform_slots = ''
        # if implicit != '':
        #     word_dict = load_dict('./data/new_dict.txt')
        #     seg = self.thu.cut(implicit)
        #     seg_list = []
        #     for s in seg:
        #         seg_list.append(s[0])
        #     for i in range(len(seg_list) - 1, -1, -1):
        #         if seg_list[i] in self.stop_words:
        #             del seg_list[i]
        #     implicit_inform_slots = replace_list(seg_list, word_dict, model=model, similarity_dict=self.similarity_dict)
        #     self.log.info(implicit_inform_slots)
        #     for i in range(len(implicit_inform_slots) - 1, -1, -1):
        #         if implicit_inform_slots[i] in self.stop_words:
        #             del implicit_inform_slots[i]
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(implicit_inform_slots,
                                                                                          agent_action=agent_action,
                                                                                          turn=self.state_tracker.turn)
        with open(self.state_tracker.user.goal_set_path, 'r') as f:
            goal_set = json.load(f)
            goal_set['user_action'] = user_action
        with open(self.state_tracker.user.goal_set_path, 'w') as f:
            json.dump(goal_set, f, indent=4, ensure_ascii=False)
        self.state_tracker.state_updater(user_action=user_action)

        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE:
            self.inform_wrong_service_count += 1

        state = self.state_tracker.get_state()

        # if user_action['action'] == 'closing':
        #     return reward, episode_over, dialogue_status, agent_action

        agent_action, action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy)
        with open(self.state_tracker.user.goal_set_path, 'r') as f:
            goal_set = json.load(f)
            goal_set['agent_action'] = agent_action
        with open(self.state_tracker.user.goal_set_path, 'w') as f:
            json.dump(goal_set, f, indent=4, ensure_ascii=False)
        self.state_tracker.state_updater(agent_action=agent_action)

        return reward, episode_over, dialogue_status, agent_action
