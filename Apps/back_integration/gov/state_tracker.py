# -*- coding:utf-8 -*-

import sys, os
import copy



class StateTracker(object):
    def __init__(self, user, agent, parameter):
        self.user = user
        self.agent = agent
        self._init()

    def initialize(self):
        self._init()

    def _init(self):
        self.turn = 0
        self.state = {
            "agent_action": None,
            "user_action": None,
            "turn": self.turn,
            "current_slots": {
                "user_request_slots": {},
                "agent_request_slots": {},
                "inform_slots": {},
                "wrong_services": []
            }
        }

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_agent(self, agent):
        self.agent = agent

    def state_updater(self, user_action=None, agent_action=None):
        self.state["turn"] = self.turn
        if user_action is not None:
            self._state_update_with_user_action(user_action=user_action)
        elif agent_action is not None:
            self._state_update_with_agent_action(agent_action=agent_action)
        self.turn += 1

    def _state_update_with_user_action(self, user_action):
        self.state["user_action"] = user_action
        for slot in user_action["request_slots"].keys():
            self.state["current_slots"]["user_request_slots"][slot] = user_action["request_slots"][slot]
        inform_slots = list(user_action["inform_slots"].keys())

        if "service" in inform_slots and user_action["action"] == "deny":
            if user_action["inform_slots"]["service"] not in self.state["current_slots"]["wrong_services"]:
                self.state["current_slots"]["wrong_services"].append(user_action["inform_slots"]["service"])
        if "service" in inform_slots:
            inform_slots.remove("service")

        for slot in inform_slots:
            if slot not in self.user.goal["request_slots"].keys():
                self.state["current_slots"]['inform_slots'][slot] = user_action["inform_slots"][slot]
            #if slot in self.state["current_slots"]["agent_request_slots"].keys():  ###这里还要想想到底要不要
                #self.state["current_slots"]["agent_request_slots"].pop(slot)

    def _state_update_with_agent_action(self, agent_action):
        self.state["agent_action"] = agent_action
        for slot in agent_action["request_slots"].keys():
            self.state["current_slots"]["agent_request_slots"][slot] = agent_action["request_slots"][slot]
        # Inform slots.
        for slot in agent_action["inform_slots"].keys():
            if slot in self.state["current_slots"]["user_request_slots"].keys():
                self.state["current_slots"]["user_request_slots"].pop(slot)
