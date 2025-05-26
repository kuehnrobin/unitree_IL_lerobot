''''
Refer to:   lerobot/configs/eval.py
'''

import logging
from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None
    
    # Speed control parameters
    arm_speed: float | None = None
    no_gradual_speed: bool = field(default=False)
    
    # Network interface for CycloneDDS
    cyclonedx_uri: str = "enxa0cec8616f27"

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )


    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
