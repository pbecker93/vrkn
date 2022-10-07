import torch
from collections import OrderedDict
from typing import Optional
import numpy as np
import random
import warnings

from ssm_mbrl.util.config_dict import ConfigDict
from ssm_mbrl.common.img_preprocessor import ImgPreprocessor


class MBRLExperiment:

    def __init__(self,
                 env_factory,
                 model_factory,
                 policy_factory,
                 trainer_factory,
                 mbrl_factory,
                 seed: int,
                 verbose: bool,
                 policy_eval_factories: Optional = None,
                 use_cuda_if_available: bool = True,
                 fully_deterministic: bool = False):

        self._verbose = verbose
        self._seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if fully_deterministic:
            warnings.warn("Fully Deterministic run requested... this will be slower!")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._conf_save_dict = {}

        self._env_factory = env_factory
        self._model_factory = model_factory
        self._policy_factory = policy_factory
        self._trainer_factory = trainer_factory
        self._mbrl_factory = mbrl_factory
        self._policy_eval_factories = policy_eval_factories

        self._built = False
        self._env = None
        self._model = None
        self._policy = None
        self._trainer = None
        self._model_evaluators = None
        self._policy_evaluators = None
        self._replay_buffer = None
        self._data_collector = None
        self._model_evaluators = None
        self._rb_kwargs_train = None
        self._rb_kwargs_val = None

    def build(self,
              env_config: ConfigDict,
              model_config: ConfigDict,
              policy_config: ConfigDict,
              mbrl_config: ConfigDict,
              trainer_config: ConfigDict,
              policy_eval_config: dict[str, ConfigDict]):

        # Env
        self._conf_save_dict["env_config"] = env_config
        if self._verbose:
            print("=== Environment ===")
            print(env_config)
        base_env = self._env_factory.build(seed=self._seed,
                                           config=env_config)

        # MBRL Stuff
        self._conf_save_dict["mbrl_config"] = mbrl_config
        if self._verbose:
            print("=== MBRL ===")
            print(mbrl_config)

        if any(base_env.obs_are_images):
            img_preprocessor = ImgPreprocessor(depth_bits=mbrl_config.img_preprocessing.color_depth_bits,
                                               add_cb_noise=mbrl_config.img_preprocessing.add_cb_noise)
        else:
            img_preprocessor = None
        self._env, self._replay_buffer = \
            self._mbrl_factory.build_env_and_replay_buffer(env=base_env,
                                                           img_preprocessor=img_preprocessor,
                                                           config=mbrl_config)
        self._rb_kwargs_train = {"seq_length": mbrl_config.mbrl_exp.model_updt_seq_length,
                                 "batch_size": mbrl_config.mbrl_exp.model_updt_batch_size,
                                 "num_batches": mbrl_config.mbrl_exp.model_updt_steps}
        self._rb_kwargs_val = {"seq_length": mbrl_config.mbrl_exp.model_updt_seq_length,
                               "batch_size": mbrl_config.validation.batch_size,
                               "num_batches": mbrl_config.validation.num_batches}

        # Model
        self._conf_save_dict["model_config"] = model_config
        if self._verbose:
            print("=== MODEL ===")
            print(model_config)

        obs_sizes = [o.shape[0] if len(o.shape) == 1 else o.shape for o in self._env.observation_space]
        self._model = self._model_factory.build(config=model_config,
                                                input_sizes=obs_sizes,
                                                output_sizes=obs_sizes + [1],
                                                action_dim=base_env.action_dim,
                                                with_obs_valid=self._replay_buffer.has_obs_valid).to(self._device)

        # Policy
        self._conf_save_dict["policy_config"] = policy_config
        if self._verbose:
            print("=== Policy ===")
            print(policy_config)

        self._policy = self._policy_factory.build(model=self._model,
                                                  obs_are_images=base_env.obs_are_images,
                                                  img_preprocessor=img_preprocessor,
                                                  config=policy_config,
                                                  action_space=base_env.action_space,
                                                  device=self._device)
        self._data_collector = self._mbrl_factory.build_data_collector(env=self._env,
                                                                       policy=self._policy,
                                                                       config=mbrl_config)

        # Trainer
        self._conf_save_dict["trainer"] = trainer_config
        if self._verbose:
            print("=== Policy Trainer ===")
            print(trainer_config)
        self._trainer = self._trainer_factory.build(policy=self._policy,
                                                    model=self._model,
                                                    config=trainer_config)

        self._policy_evaluators = []
        if self._policy_eval_factories is not None:
            if policy_eval_config is None:
                policy_eval_config = ConfigDict()
            for factory in self._policy_eval_factories:
                config = policy_eval_config[factory.name()]
                evaluator = factory.build(env=self._env,
                                          policy=self._policy,
                                          config=config)
                if self._verbose:
                    print("=== {} ===".format(evaluator.name()))
                    print(config)
                self._policy_evaluators.append(evaluator)
            self._conf_save_dict["policy_eval"] = policy_eval_config
        self._built = True

    def iterate(self, iteration: int):

        assert self._built
        if self._verbose:
            print("=== Iteration {:04d} ===".format(iteration))

        train_loader = self._replay_buffer.get_data_loader(device=self._device, **self._rb_kwargs_train)
        train_dict, train_time = self._trainer.train_epoch(data_loader=train_loader)
        with torch.no_grad():
            log_dict = self._model_log(model_log_dict=train_dict,
                                       time=train_time)

            log_dict, val_loader = self._eval_model(iteration=iteration,
                                                    log_dict=log_dict)
            log_dict = self._collect_new_data(iteration=iteration,
                                              log_dict=log_dict)
            log_dict = self._eval_policy(iteration=iteration,
                                         log_dict=log_dict)
            return log_dict

    def _eval_model(self,
                    iteration: int,
                    log_dict: OrderedDict):
        if self._trainer.will_evaluate(iteration):
            val_loader = self._replay_buffer.get_data_loader(device=self._device,
                                                             **self._rb_kwargs_val)
        else:
            val_loader = None
        val_dict = self._trainer.evaluate(data_loader=val_loader, iteration=iteration)
        log_dict = self._model_log(model_log_dict=val_dict,
                                   time=None,
                                   training=False,
                                   log_dict=log_dict)
        if self._model_evaluators is not None:
            eval_results = []
            for evaluator in self._model_evaluators:
                eval_results.append(evaluator.evaluate(data_loader=val_loader,
                                                       iteration=iteration))
            log_dict = self._log_evaluators(eval_results=eval_results,
                                            evaluators=self._model_evaluators,
                                            log_dict=log_dict)
            return log_dict, val_loader
        else:
            return log_dict, None

    def _eval_policy(self,
                     iteration: int,
                     log_dict: OrderedDict):
        if self._policy_evaluators is not None:
            eval_results = []
            for evaluator in self._policy_evaluators:
                eval_results.append(evaluator.evaluate(iteration=iteration))
            log_dict = self._log_evaluators(eval_results=eval_results,
                                            evaluators=self._policy_evaluators,
                                            log_dict=log_dict)
        return log_dict

    def _collect_new_data(self,
                          iteration: int,
                          log_dict: OrderedDict):

        assert not self._replay_buffer.is_frozen
        observations, actions, rewards, infos, collect_time = self._data_collector.collect()
        avg_reward = sum([torch.sum(r).detach().numpy() for r in rewards]) / len(rewards)
        avg_seq_length = sum([len(r) for r in rewards]) / len(rewards)
        log_dict = self._log_collection(num_seqs=len(rewards),
                                        avg_reward=avg_reward,
                                        avg_seq_length=avg_seq_length,
                                        collect_time=collect_time,
                                        log_dict=log_dict)

        self._replay_buffer.add_data(key="iter{:04d}".format(iteration),
                                     observations=observations,
                                     actions=actions,
                                     rewards=rewards,
                                     infos=infos)
        return log_dict

    def _model_log(self,
                   model_log_dict: dict,
                   time: Optional[float] = None,
                   log_dict: Optional[OrderedDict] = None,
                   training=True) -> OrderedDict:
        prefix, long_str = ("train", "Training") if training else ("eval", "Validation")
        if self._verbose and len(model_log_dict) > 0:
            log_str = "Model {}, ".format(long_str)
            for k, v in model_log_dict.items():
                if np.isscalar(v):
                    log_str += "{}: {:.5f} ".format(k, v)
            if time is not None:
                log_str += "Took {:.3f} seconds".format(time)
            print(log_str)
        if log_dict is None:
            log_dict = OrderedDict()
        for k, v in model_log_dict.items():
            if "/" in k:
                log_dict[k.replace("/", "_{}/".format(prefix))] = v
            else:
                log_dict["{}/{}".format(prefix, k)] = v
        if time is not None:
            log_dict["{}/time".format(prefix)] = time
        return log_dict

    def _log_evaluators(self,
                        eval_results: list,
                        evaluators: list,
                        log_dict: Optional[OrderedDict] = None) -> OrderedDict:
        if eval_results is not None:
            if self._verbose:
                for results, evaluator in zip(eval_results, evaluators):
                    log_str = "{}: ".format(evaluator.name())
                    if results is not None:
                        for k, v in results.items():
                            log_str += self.log_str_form_kv_pair(k, v)
                        print(log_str)
            if log_dict is None:
                log_dict = OrderedDict()
            for results, evaluator in zip(eval_results, evaluators):
                if results is not None:
                    for k, v in results.items():
                        log_dict["{}/{}".format(evaluator.name(), k)] = v
            return log_dict

    @staticmethod
    def log_str_form_kv_pair(k, v) -> str:
        if isinstance(v, float):
            log_str = "{}: {:.5f} ".format(k, v)
        elif isinstance(v, np.ndarray):
            log_str = "{}: {} ".format(k,
                                       np.array2string(v, precision=5, max_line_width=int(1e300)))
        else:
            log_str = "{}: {} ".format(k, str(v))
        return log_str

    def _log_collection(self,
                        num_seqs: int,
                        avg_reward: float,
                        avg_seq_length: float,
                        collect_time: float,
                        log_dict: Optional[OrderedDict] = None) -> OrderedDict:
        if self._verbose:
            collect_log_str = "Data Collection: Collected {:03d} Sequence(s) ".format(num_seqs)
            collect_log_str += "with average reward of {:.5f} ".format(avg_reward)
            collect_log_str += "and average length of {:.2f} ".format(avg_seq_length)
            collect_log_str += "Took {:.3f} seconds.".format(collect_time)
            print(collect_log_str)
        if log_dict is None:
            log_dict = OrderedDict()
        log_dict["collect/num_seqs"] = num_seqs
        log_dict["collect/avg_len"] = avg_seq_length
        log_dict["collect/avg_reward"] = avg_reward
        log_dict["collect/time"] = collect_time
        return log_dict

