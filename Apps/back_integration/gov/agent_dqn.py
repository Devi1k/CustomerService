# -*-coding:utf-8 -*

import random

from Apps.back_integration.gov.agent import Agent
from Apps.back_integration.gov.dqn_torch import DQN2 as DQN


class AgentDQN(Agent):
    def __init__(self, parameter):
        super(AgentDQN, self).__init__(parameter=parameter)
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.action_space)
        self.dqn = DQN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, parameter=parameter)

    def next(self, state, turn, greedy_strategy):  # greedy_strategy：train_mode int, 1:for training, 0:for evaluation
        self.agent_action["turn"] = turn
        state_rep = self.state_to_representation_last(state=state)  # sequence representation.🔺

        if greedy_strategy == 1:
            greedy = random.random()
            if greedy < self.parameter.get(
                    "epsilon"):  # ("--epsilon", dest="epsilon", type=float, default=0.1, help="the greedy of DQN")
                action_index = random.randint(0, len(self.action_space) - 1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]

        agent_action = self.action_space[action_index]

        agent_action["turn"] = turn
        return agent_action, action_index

    def train(self, batch):
        loss = self.dqn.singleBatch(batch=batch, params=self.parameter)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()

    # 保存模型
    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        self.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index,
                            checkpoint_path=checkpoint_path)
