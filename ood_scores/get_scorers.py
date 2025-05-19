import os, sys
if __package__ is None:
    print(sys.path)
    sys.path.append(os.path.dirname(__file__))

from .scorer_dist import WDiscOOD, Maha
from .scorer_baselines import MSPScorer,  EnergyScorer, MaxLogitScorer, KLMatchScorer 
from .scorer_knn import KNN
from .scorer_maha import Mahalanobis
from .scorer_vim import VIM, Residual
from .scorer_react import ReAct
from .scorer_odin import ODIN
from .score_dice import DICE
from .score_line import Line
from .score_fdb import FDB
from .score_dynamic import Dynamic
from .score_neco import Neco
from .score_ssd import SSD

SCORER_CODEBOOK = {
    "mahavanilla": Maha,
    "Maha": Mahalanobis,
    "odin": ODIN,
    "wdiscood": WDiscOOD,
    "msp": MSPScorer,
    "energy":EnergyScorer,
    "maxlogit": MaxLogitScorer,
    "klmatch": KLMatchScorer,
    "knn": KNN,
    "vim": VIM,
    "react": ReAct,
    "residual": Residual,
    'diceood': DICE,
    'dynamic': Dynamic,
    'line': Line,
    'fdb':FDB,
    'neco': Neco,
    'ssd': SSD
}


def get_scorer(score, args):
    # print(score.lower())
    try:
        return SCORER_CODEBOOK[score.lower()](args)
    except:
        raise NotImplementedError(f"{score.lower()} not implemented. Acceptable score names: {SCORER_CODEBOOK.keys()}")