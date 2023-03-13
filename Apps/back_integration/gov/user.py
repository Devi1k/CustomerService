# -*- coding:utf-8 -*-
import json
import os.path

import Apps.back_integration.gov.dialogue_configuration as dialogue_configuration


# import gov.goal_set as goal_set


class User(object):
    def __init__(self, parameter, conv_id):
        self.max_turn = parameter["max_turn"]
        self.parameter = parameter
        self.allow_wrong_service = parameter.get("allow_wrong_service")
        self.conv_id = conv_id
        self._init()

    def _init(self):
        self.state = {
            "turn": 0,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "user_judge": True,
            "explicit_inform_slots": {},
            "implicit_inform_slots": {}
        }

        # self.goal_set = random.choice(goal_set.goal_set)
        goal = {
            "goal": {
                "request_slots": {
                    "service": "UNK"
                },
                "max_slot": {},
                "service_tag": {},
                "explicit_inform_slots": {},
                "implicit_inform_slots": {}
            },
            "user_action": {},
            "agent_aciton": {}
        }
        self.goal_set = goal
        self.goal_set_path = os.path.join('./data', 'goal_set_{}.json'.format(self.conv_id))
        with open(self.goal_set_path, 'w') as f:
            json.dump(goal, f)
        self.goal = self.goal_set["goal"]
        self.episode_over = False
        self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET

    def initialize(self, _explicit_inform_slots):
        self._init()
        self.state["action"] = "request"
        self.state["request_slots"]["service"] = dialogue_configuration.VALUE_UNKNOWN
        explicit_inform_slots = dict()
        for i in _explicit_inform_slots:
            explicit_inform_slots[i] = True
        self.goal["explicit_inform_slots"] = explicit_inform_slots
        for slot in self.goal["explicit_inform_slots"].keys():
            self.state["inform_slots"][slot] = self.goal["explicit_inform_slots"][slot]
        user_action = self._assemble_user_action()
        return user_action

    def _assemble_user_action(self):
        user_action = {
            "turn": self.state["turn"],
            "action": self.state["action"],
            "speaker": "user",
            "request_slots": self.state["request_slots"],
            "inform_slots": self.state["inform_slots"],
            "explicit_inform_slots": self.state["explicit_inform_slots"],
            "implicit_inform_slots": self.state["implicit_inform_slots"],
            "user_judge": self.state["user_judge"]
        }
        return user_action

    def next(self, _implicit_inform_slots, agent_action, turn):
        implicit_inform_slots = dict()
        if _implicit_inform_slots != '':
            for i in _implicit_inform_slots:
                implicit_inform_slots[i] = True

        self.goal['implicit_inform_slots'] = implicit_inform_slots
        agent_act_type = agent_action["action"]
        self.state["turn"] = turn
        if self.state["turn"] == self.max_turn - 2:  # 这里还要改
            self.episode_over = True
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
            pass

        if self.episode_over is not True:
            if agent_act_type == "request":
                self._response_request(agent_action=agent_action)
            elif agent_act_type == "inform":
                self._response_inform(agent_action=agent_action)
            user_action = self._assemble_user_action()
            reward = self._reward_function()
            return user_action, reward, self.episode_over, self.dialogue_status
        else:
            if _implicit_inform_slots == '':
                if agent_act_type == "request":
                    self._response_request(agent_action=agent_action)
                elif agent_act_type == "inform":
                    self._response_inform(agent_action=agent_action)
            user_action = self._assemble_user_action()
            reward = self._reward_function()
            return user_action, reward, self.episode_over, self.dialogue_status

    def _response_request(self, agent_action):
        """
        把agent_action["request_slots"]传到前端
        :param agent_action:
        """
        with open(self.goal_set_path, 'r') as f:
            goal_set = json.load(f)
        slots = agent_action["request_slots"].keys()
        for slot in slots:
            if goal_set['user_action']['user_judge'] is True:
                self.state["user_judge"] = goal_set['user_action']['user_judge']
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = True
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SLOT
            elif goal_set['user_action']['user_judge'] is False:
                self.state["user_judge"] = goal_set['user_action']['user_judge']
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = False
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SLOT
                for slot in self.goal["implicit_inform_slots"].keys():
                    self.state["inform_slots"][slot] = self.goal["implicit_inform_slots"][slot]

    def _response_inform(self, agent_action):
        """
        把agent_action["inform_slots"]["service"]传到前端
        :param agent_action:
        """
        # agent_all_inform_slots = copy.deepcopy(agent_action["inform_slots"])
        # user_all_inform_slots = copy.deepcopy(self.goal["explicit_inform_slots"])
        # user_all_inform_slots.update(self.goal["implicit_inform_slots"])

        with open(self.goal_set_path, 'r') as f:
            goal_set = json.load(f)
        if "service" in agent_action["inform_slots"] and goal_set['user_action']['user_judge'] is True:
            self.state["user_judge"] = goal_set['user_action']['user_judge']
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_SUCCESS
            self.episode_over = True
            self.state["inform_slots"].clear()
            self.state["request_slots"].pop("service")
        elif "service" in agent_action["inform_slots"] and goal_set['user_action']['user_judge'] is not True:
            self.state["user_judge"] = goal_set['user_action']['user_judge']
            if self.allow_wrong_service == 1:
                self.state["action"] = "deny"
                # 3.20 lyj 改  增145，146  删147  wrong_service加进去，用于agent计算score
                service_name = agent_action["inform_slots"]["service"]
                self.state["inform_slots"][service_name] = False
                # self.state["inform_slots"]["service"] = agent_action["inform_slots"]["service"]
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE
                # 3.20 lyj  用户新增信息也要加进去，同request部分
                for slot in self.goal["implicit_inform_slots"].keys():
                    self.state["inform_slots"][slot] = self.goal["implicit_inform_slots"][slot]
            else:
                self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
                self.dialogue_status = dialogue_configuration.DIALOGUE_STATUS_FAILED
                self.episode_over = True
                self.state["inform_slots"].clear()

    def _reward_function(self):  # 从这里可以看到所有的dialogue_status
        if self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET:
            return dialogue_configuration.REWARD_FOR_NOT_COME_YET
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_SUCCESS
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_FAILED:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_FAILED
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SERVICE:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_SERVICE
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_RIGHT_SLOT:
            return dialogue_configuration.REWARD_FOR_INFORM_RIGHT_SLOT
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_SLOT:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_SLOT
