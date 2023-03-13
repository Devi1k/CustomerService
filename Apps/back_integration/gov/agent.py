# -*- coding: utf-8 -*-
import copy
import json
import os.path

import numpy as np

import Apps.back_integration.gov.dialogue_configuration as dialogue_configuration
from Apps.back_integration.gov.slot_config import requirement_weight, service


# from govBackEnd.gov.slot_config import slot_max_weight, slot_set


class Agent(object):
    def __init__(self, parameter):
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_path, 'data/slot_set.json'), 'r') as f:
            slot_set = json.load(f)
        with open(os.path.join(root_path, 'data/slot_max_weight.json'), 'r') as f:
            slot_max_weight = json.load(f)
        self.slot_max_weight = slot_max_weight
        self.slot_set = slot_set
        self.slot_max = list(self.slot_max_weight.keys())
        self.action_set = {'request': 0, 'inform': 1, 'closing': 2, 'deny': 3}
        self.service = service
        self.parameter = parameter
        self.action_space = self._build_action_space()
        self.agent_action = {
            "turn": 1,
            "action": None,
            "speaker": "agent",
            "request_slots": {},
            "inform_slots": {}
        }

    def initialize(self):
        self.agent_action = {
            "turn": None,
            "action": None,
            "speaker": "agent",
            "request_slots": {},
            "inform_slots": {}
        }
        self.requirement_weight = copy.deepcopy(requirement_weight)
        self.service = copy.deepcopy(service)
        # self.slot_max = copy.deepcopy(slot_max)

    def _build_action_space(self):  # warm_start没用到，dqn部分用
        feasible_actions = [
            {'action': dialogue_configuration.CLOSE_DIALOGUE, 'inform_slots': {}, 'request_slots': {}},
            {'action': dialogue_configuration.THANKS, 'inform_slots': {}, 'request_slots': {}}
        ]
        #   Adding the inform actions and request actions.
        for slot in sorted(self.slot_max_weight.keys()):
            feasible_actions.append({'action': 'request', 'inform_slots': {},
                                     'request_slots': {slot: dialogue_configuration.VALUE_UNKNOWN}})
        # Services as actions.
        for slot in self.service:
            feasible_actions.append({'action': 'inform', 'inform_slots': {"service": slot}, 'request_slots': {}})

        return feasible_actions

    def state_to_representation_last(self, state):
        current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
        current_slots_rep = np.zeros(len(self.slot_set.keys()))
        for slot in current_slots.keys():
            # 用权重
            # current_slots_rep[self.slot_set[slot]] =current_slots[slot]
            if current_slots[slot] == True:
                current_slots_rep[self.slot_set[slot]] = 1.0
            elif current_slots[slot] == False:
                current_slots_rep[self.slot_set[slot]] = -1.0
        turn_rep = np.zeros(self.parameter["max_turn"] + 1)
        turn_rep[state["turn"]] = 1.0
        user_action_rep = np.zeros(len(self.action_set))
        user_action_rep[self.action_set[state["user_action"]["action"]]] = 1.0
        user_inform_slots = copy.deepcopy(state["user_action"]["inform_slots"])
        user_inform_slots.update(state["user_action"]["explicit_inform_slots"])
        user_inform_slots.update(state["user_action"]["implicit_inform_slots"])
        if "service" in user_inform_slots: user_inform_slots.pop("service")
        user_inform_slots_rep = np.zeros(len(self.slot_set.keys()))
        for slot in user_inform_slots.keys():
            # user_inform_slots_rep[self.slot_set[slot]] = user_inform_slots[slot]
            if user_inform_slots[slot] == True:
                user_inform_slots_rep[self.slot_set[slot]] = 1.0
            elif user_inform_slots[slot] == False:
                user_inform_slots_rep[self.slot_set[slot]] = -1.0
        user_request_slots = copy.deepcopy(state["user_action"]["request_slots"])
        user_request_slots_rep = np.zeros(len(self.slot_set.keys()))
        for slot in user_request_slots.keys():
            user_request_slots_rep[self.slot_set[slot]] = 1.0
        agent_action_rep = np.zeros(len(self.action_set))
        try:
            agent_action_rep[self.action_set[state["agent_action"]["action"]]] = 1.0
        except:
            pass
        agent_inform_slots_rep = np.zeros(len(self.slot_set.keys()))
        try:
            agent_inform_slots = copy.deepcopy(state["agent_action"]["inform_slots"])
            for slot in agent_inform_slots.keys():
                agent_inform_slots_rep[self.slot_set[slot]] = 1.0
        except:
            pass
        agent_request_slots_rep = np.zeros(len(self.slot_set.keys()))
        try:
            agent_request_slots = copy.deepcopy(state["agent_action"]["request_slots"])
            for slot in agent_request_slots.keys():
                agent_request_slots_rep[self.slot_set[slot]] = 1.0
        except:
            pass
        state_rep = np.hstack((current_slots_rep, user_action_rep, user_inform_slots_rep,
                               user_request_slots_rep, agent_action_rep, agent_inform_slots_rep,
                               agent_request_slots_rep, turn_rep))
        return state_rep
