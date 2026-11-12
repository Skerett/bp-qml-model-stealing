from enum import Enum

class ApiMode(str, Enum):
    CONFIDENCE_SCORE = "confidence_score"
    HARD_LABEL = "hard_label"
    ROUNDED = "rounded"
    NOISY = "noisy"
