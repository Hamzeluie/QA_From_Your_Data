import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import dspy
from config.settings import settings

model_id = "openai/" + settings.LLM_MODEL_NAME

lm = dspy.LM(
    model=model_id,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    max_tokens=4096,
    temperature=0.0,
    top_p=1.0,
    n=1,
    stop=None
)
dspy.configure(lm=lm)

