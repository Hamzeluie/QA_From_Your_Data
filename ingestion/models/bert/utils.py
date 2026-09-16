import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from typing import Optional
from transformers import AutoTokenizer


def _is_valid_local_model(path: str) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    return (p / "config.json").exists() or (p / "tokenizer_config.json").exists()


def _load_tokenizer_robust(load_path: str, model_name: str, local_dir: Optional[str]):
    kwargs = {"use_fast": True, "trust_remote_code": True}

    if _is_valid_local_model(load_path):
        try:
            return AutoTokenizer.from_pretrained(load_path, local_files_only=True, **kwargs)
        except Exception as e:
            print(f"[warn] Local tokenizer at {load_path} failed: {e}")

    print(f"[info] Downloading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, **kwargs)

    if local_dir:
        save_path = Path(local_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(save_path)
        print(f"[info] Tokenizer cached to {save_path}")

    return tok


def _load_model_robust(
    load_path: str,
    model_name: str,
    local_dir: Optional[str],
    onnx_path: Optional[Path],
    use_onnx: bool,
    model_cls,
    ort_cls,
    resize_tokens: int = 0,
):
    # 1. Existing ONNX
    if use_onnx and onnx_path and (onnx_path / "model.onnx").exists():
        if not (onnx_path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json missing from {onnx_path}. "
                f"Copy it from your trained checkpoint."
            )
        print(f"[info] Loading ONNX from {onnx_path}")
        return ort_cls.from_pretrained(str(onnx_path)), True

    # 2. Local PyTorch checkpoint
    if _is_valid_local_model(load_path):
        try:
            if use_onnx:
                print(f"[info] Exporting local model to ONNX: {load_path}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ort_cls.from_pretrained(load_path, export=True)
                if onnx_path:
                    onnx_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(onnx_path))
                return model, True
            else:
                model = model_cls.from_pretrained(load_path, local_files_only=True)
                model.eval()
                return model, True
        except Exception as e:
            print(f"[warn] Local model load failed: {e}")

    # 3. Download from Hub
    print(f"[info] Downloading model: {model_name}")
    model = model_cls.from_pretrained(model_name)
    model.eval()

    if resize_tokens > 0 and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(resize_tokens)

    if local_dir:
        save_path = Path(local_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        print(f"[info] Model cached to {save_path}")

    if use_onnx and onnx_path:
        onnx_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(onnx_path))
        print(f"[info] ONNX saved to {onnx_path}")
        return ort_cls.from_pretrained(str(onnx_path)), False

    return model, False

