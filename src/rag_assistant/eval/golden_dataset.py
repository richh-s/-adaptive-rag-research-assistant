import json
from pathlib import Path

from rag_assistant.config import PROJECT_ROOT
from rag_assistant.schemas.models import GoldenQuestion

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "golden_eval" / "dataset.jsonl"


def load_golden_dataset(path: Path | None = None) -> list[GoldenQuestion]:
    dataset_path = path or DEFAULT_DATASET_PATH
    questions = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(GoldenQuestion.model_validate(json.loads(line)))
    return questions
