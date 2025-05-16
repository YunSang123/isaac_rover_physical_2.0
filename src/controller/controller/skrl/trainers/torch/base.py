from typing import List, Optional, Union

import atexit
import sys
import tqdm
import inspect
import torch

from skrl import config, logger
from skrl.agents.torch import Agent
# Agent 클래스는 /isaac-sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/base.py에서 불러와짐.
# print(f"Agent 경로 = {inspect.getfile(Agent)}\n"*10)
from skrl.envs.wrappers.torch import Wrapper

from .data_recorder.hdf_recorder import HDF5DataRecorder


def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})")
    return scopes


class Trainer:
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        # print(f"self.agents = {self.agents}\n"*20)
        self.agents_scope = agents_scope if agents_scope is not None else []
        self.recorder = HDF5DataRecorder(base_filename="best_agent", num_envs = 1, env = self.env)

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")

        self.initial_timestep = 0

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:
            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})")
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})")
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})")
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        # print("isaac_rover/skrl/trainers/torch/base.py에서 eval 실행중!")
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    # 실제로 실행되는 함수
    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"
        
        self.state_history = []

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):
            # pre-interaction
            """
            - 실제로 실행되는 함수임.
            - 에이전트가 상호작용 전에 준비 작업을 수행
            """
            # pre_interaction 실행 경로 = /isaac-sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/ppo/ppo.py
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # 사실상 아무것도 안하긴함.

            # compute actions
            # with torch.no_grad() = 그래디언트 계산을 비활성화하는 것임.
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                # print(f"actions = {actions}")
                # print("states\n", states)
                
                # 상태 저장
                # self.state_history.append(states.clone())  # clone() 사용해 복사본 저장
                
                # print("isaac_rover/skrl/trainers/torch/base.py")
                # print(type(self.env))  # 현재 self.env가 어떤 클래스의 인스턴스인지 확인
                # print(self.env.__class__.__mro__)  # 클래스의 상속 계층 구조 확인
                # print("isaac_rover/skrl/trainers/torch/base.py")
                
                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states
        # torch.save(self.state_history, "state_history.pth")  # PyTorch 텐서를 저장
        
    def single_agent_record(self) -> None:
        """Record agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # pre-interaction
            """
            - 실제로 실행되는 함수임.
            - 에이전트가 상호작용 전에 준비 작업을 수행
            """
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            # with torch.no_grad() = 그래디언트 계산을 비활성화하는 것임.
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                print(f"actions : {actions}")
                
                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()
                    
                self.recorder.append_to_buffer(states, actions, rewards, terminated, infos)
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states

                # # record the environments' transitions
                # self.agents.record_transition(states=states,
                #                               actions=actions,
                #                               rewards=rewards,
                #                               next_states=next_states,
                #                               terminated=terminated,
                #                               truncated=truncated,
                #                               infos=infos,
                #                               timestep=timestep,
                #                               timesteps=self.timesteps)

                # # log environment info
                # if self.environment_info in infos:
                #     for k, v in infos[self.environment_info].items():
                #         if isinstance(v, torch.Tensor) and v.numel() == 1:
                #             self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            # self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states
        # torch.save(self.state_history, "state_history.pth")  # PyTorch 텐서를 저장

    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # print("assert 실행하기 전!\n"*10)
        
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # print("single_agent_eval에서 self.env.reset() 실행되기 전!")
        # reset env
        states, infos = self.env.reset()
        # print("single_agent_eval에서 self.env.reset() 실행된 후!")

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):
            
            # print("skrl/trainers/torch/base.py에서 실행중\n"*10)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                # print(f"actions = {actions}")

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def multi_agent_train(self) -> None:
        """Train multi-agents

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            with torch.no_grad():
                if not self.env.agents:
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
                else:
                    states = next_states
                    shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # print("multi_agent_eval 실행중!")
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout):

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

                # reset environments
                if not self.env.agents:
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
                else:
                    states = next_states
                    shared_states = shared_next_states
