"""Single-device PyTorch implementations of optimizers compared against a tuned
AdamW baseline for multi-epoch pretraining on a fixed token budget."""
from .adamhd import AdamHD
from .adaml1 import AdamL1
from .factory import DEFAULTS, OPTIMIZERS, build_optimizer
from .hyperball import AdamHyperball, MuonEqHyperball, hyperball_step_
from .muoneq import MuonEqAdamW, equilibrate, zeropower_via_newtonschulz5
from .nexus import Nexus
from .param_groups import split_body_params
from .sam import SAM
from .soap import SOAPAdamW, SOAPHyperball

__all__ = [
    "AdamHD", "AdamL1", "AdamHyperball", "MuonEqAdamW", "MuonEqHyperball",
    "Nexus", "SAM", "SOAPAdamW", "SOAPHyperball",
    "build_optimizer", "split_body_params", "equilibrate",
    "zeropower_via_newtonschulz5", "hyperball_step_", "DEFAULTS", "OPTIMIZERS",
]
__version__ = "0.1.0"
