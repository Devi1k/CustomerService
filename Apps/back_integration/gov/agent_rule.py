# -*- coding: utf-8 -*-

import copy

import Apps.back_integration.gov.dialogue_configuration as dialogue_configuration
from Apps.back_integration.gov.agent import Agent


class AgentRule(Agent):
    def __init__(self, parameter):
        super(AgentRule, self).__init__(parameter=parameter)
        # self.requirement_weight=requirement_weight  #为什么这种不行啊
        # self.service=service
        # self.slot_max=slot_max

    def next(self, state, turn, greedy_strategy):
        score_max = 0
        max = 0
        # delete = 0
        score = []
        for i in range(len(self.slot_max)):
            score.append(0)
        # pop掉wrong max_slot
        # print(state["current_slots"]["agent_request_slots"].keys()) #为什么这里是空啊
        # 3.20 lyj改
        """
        if state["user_action"]["user_judge"] == False:
            for i in range(len(self.slot_max)):
                # 每次pop掉最后一个，即新增的那个
                request_slots_list = list(state["current_slots"]["agent_request_slots"].keys())
                if self.slot_max[i] == request_slots_list[
                    len(state["current_slots"]["agent_request_slots"].keys()) - 1]:
                    delete = i  # 记下位置
            state["user_action"]["inform_slots"].pop(self.slot_max[delete])
            self.slot_max.pop(delete)
            self.requirement_weight.pop(delete)
            self.service.pop(delete)

        inform_slots = list(state["current_slots"]["inform_slots"].keys())

        if len(list(state["user_action"]["inform_slots"].keys())) == 0:  # 即用户没有提供信息
            if state["user_action"]["user_judge"] == False:  # 且否定了
                max = randint(0, len(self.slot_max) - 1)  # 就从剩下的随机选一个
        else:
            for i in range(len(self.slot_max)):
                for j in range(len(inform_slots)):
                    if inform_slots[j] in self.requirement_weight[
                        i].keys():  # 如果该事项包含的slots中有inform的slot，该事项的score就加上该slot的权重
                        if state["current_slots"]["inform_slots"][inform_slots[j]] == True:
                            score[i] += self.requirement_weight[i][inform_slots[j]]
                        elif state["current_slots"]["inform_slots"][inform_slots[j]] == False:
                            score[i] -= self.requirement_weight[i][inform_slots[j]]
                    else:
                        pass
                if score[i] > score_max:  # 然后比较score的值，选出最大的那个，用max记下score最大的位置
                    score_max = score[i]
                    max = i
        """
        #
        inform_slots = list(state["current_slots"]["inform_slots"].keys())
        for i in range(len(self.slot_max)):
            for inform in inform_slots:
                if inform in self.requirement_weight[i].keys():
                    if state["current_slots"]["inform_slots"][inform] is True:
                        score[i] += self.requirement_weight[i][inform]
                    elif state["current_slots"]["inform_slots"][inform] is False:
                        score[i] -= self.requirement_weight[i][inform]
                else:
                    pass
            if score[i] > score_max:  # 然后比较score的值，选出最大的那个，用max记下score最大的位置
                score_max = score[i]
                max = i
        candidate_service = self.service[max]
        candidate_requirement = self.slot_max[max]
        print(candidate_requirement,max)
        self.agent_action["request_slots"].clear()
        self.agent_action["inform_slots"].clear()
        self.agent_action["turn"] = turn

        if score_max >= 100:
            self.agent_action["action"] = "inform"
            self.agent_action["inform_slots"]["service"] = candidate_service
        else:
            requirement = candidate_requirement
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"][requirement] = dialogue_configuration.VALUE_UNKNOWN
        agent_action = copy.deepcopy(self.agent_action)
        agent_action.pop("turn")
        agent_action.pop("speaker")
        agent_index = self.action_space.index(agent_action)
        return self.agent_action, agent_index
