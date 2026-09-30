from .arrays import *
from .bc_evaluator import BCEvaluator
from .bc_training import *
from .colab import *
from .config import *
from .data_encoder import *
from .evaluator import MADEvaluator
try:
    from .mamujoco_rendering import MAMuJoCoRenderer
except Exception:
    # Optional MA-MuJoCo renderer; do not block MPE/SMAC imports.
    MAMuJoCoRenderer = None
try:
    from .mpe_rendering import MPERenderer
except Exception:
    MPERenderer = None
from .offline_evaluator import MADOfflineEvaluator
from .progress import *
from .rendering import *
from .serialization import *
from .setup import *
from .smac_rendering import SMACRenderer
from .training import *


def __getattr__(name):
    # MPE and SMAC must not import the optional MuJoCo renderer.
    if name == "MAHalfCheetahRenderer":
        from .mahalfcheetah_rendering import MAHalfCheetahRenderer
        globals()[name] = MAHalfCheetahRenderer
        return MAHalfCheetahRenderer
    raise AttributeError(name)
