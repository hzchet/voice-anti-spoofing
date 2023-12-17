import importlib
import logging
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.logger import setup_logging
from src.utils import write_yaml, ROOT_PATH


@hydra.main(config_path='/workspace/configs', config_name="config.yaml", version_base="1.3")
class ConfigParser:
    def __init__(self, config: DictConfig, resume=None, finetune=None, modification=None, run_id=None):
        """
        class to parse configuration yaml file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param config: omegaconf.DictConfig containing configurations, hyperparameters for training.
                       contents of `config.yaml` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict {keychain: value}, specifying position values to be replaced
                             from config dict.
        :param run_id: Unique Identifier for training processes.
                       Used to save checkpoints and training log. Timestamp is being used as default
        """
        self._config = self._update_config(config, modification)
        self.resume = resume
        self.finetune = finetune
        
        if "trainer" in self.config:
            # set save_dir where trained model and log will be saved.
            save_dir = ROOT_PATH / self.config["trainer"]["save_dir"]

            exper_name = self.config["name"]
            if run_id is None:  # use timestamp as default run-id
                run_id = datetime.now().strftime(r"%m%d_%H%M%S")
            self._save_dir = str(save_dir / "models" / exper_name / run_id)
            self._log_dir = str(save_dir / "log" / exper_name / run_id)

            # make directory for saving checkpoints and log.
            exist_ok = run_id == ""
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

            # save updated config file to the checkpoint dir
            write_yaml(OmegaConf.to_container(self.config), self.save_dir / "config.yaml")
            # configure logging module            
            setup_logging(self.log_dir)
        else:
            setup_logging()
    
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    
    @staticmethod
    def init_obj(obj_dict, default_module=None, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj(config['param'], module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if hasattr(obj_dict, 'module'):
            default_module = importlib.import_module(obj_dict.module)

        # print(obj_dict)
        
        module_name = obj_dict.type
        module_args = dict(obj_dict.args)
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(default_module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name].type
        module_args = dict(self[name].args)
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
    
    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]
    
    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return Path(self._save_dir)

    @property
    def log_dir(self):
        return Path(self._log_dir)

    @classmethod
    def get_default_configs(cls):
        config_path = ROOT_PATH / "configs" / "config.yaml"
        with config_path.open() as f:
            return cls(OmegaConf.load(f))

    def _update_config(self, config, modification):
        if modification is None:
            return config

        for k, v in modification.items():
            if v is not None:
                self._set_by_path(config, k, v)
        return config

    def _set_by_path(self, tree, keys, value):
        """Set a value in a nested object in tree by sequence of keys."""
        keys = keys.split(";")
        self._get_by_path(tree, keys[:-1])[keys[-1]] = value
    
    @staticmethod
    def _get_by_path(tree, keys):
        """Access a nested object in tree by sequence of keys."""
        return reduce(getitem, keys, tree)
