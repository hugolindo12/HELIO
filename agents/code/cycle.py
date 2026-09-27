"""
HEILO Code Agent Controlled Cycle
"""
from enum import Enum, auto


class CyclePhase(Enum):
    UNDERSTAND = auto()
    PLAN = auto()
    INSPECT = auto()
    IDENTIFY = auto()
    PROPOSE = auto()
    APPLY = auto()
    TEST = auto()
    ANALYZE = auto()
    FIX = auto()
    FINISH = auto()
    ABORT = auto()


PHASE_ORDER = [
    CyclePhase.UNDERSTAND,
    CyclePhase.PLAN,
    CyclePhase.INSPECT,
    CyclePhase.IDENTIFY,
    CyclePhase.PROPOSE,
    CyclePhase.APPLY,
    CyclePhase.TEST,
    CyclePhase.ANALYZE,
]


def next_phase(current: CyclePhase, test_passed: bool = None, attempts: int = 0, max_attempts: int = 5) -> CyclePhase:
    if current == CyclePhase.ANALYZE:
        if test_passed is True:
            return CyclePhase.FINISH
        if attempts >= max_attempts:
            return CyclePhase.ABORT
        return CyclePhase.FIX
    if current == CyclePhase.FIX:
        return CyclePhase.APPLY
    try:
        idx = PHASE_ORDER.index(current)
        if idx + 1 < len(PHASE_ORDER):
            return PHASE_ORDER[idx + 1]
    except ValueError:
        pass
    return CyclePhase.FINISH
