import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig


def has_any_weight_files(d: Path) -> bool:
    return any(d.glob("*.safetensors")) or any(d.glob("*.bin"))


def main():
    parser = argparse.ArgumentParser(description="Check model folder validity; run vLLM test if CUDA+vLLM available.")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    print(f"[INFO] model_dir = {model_dir}")

    # 1) 기본 파일 체크 (CUDA 없어도 가능)
    if not (model_dir / "config.json").exists():
        raise RuntimeError("config.json이 없습니다.")
    if not has_any_weight_files(model_dir):
        raise RuntimeError("weight 파일(*.safetensors/*.bin)이 없습니다.")
    if not (model_dir / "chat_template.jinja").exists():
        print("[WARN] chat_template.jinja가 없습니다. (대회 템플릿 요구사항이면 포함 권장)")

    # 2) 토크나이저/컨피그 로딩 체크
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    print("[OK] transformers 로딩 성공 (config/tokenizer)")
    print(f"[INFO] model_type={getattr(cfg, 'model_type', None)}")
    print(f"[INFO] chat_template_set={tok.chat_template is not None}")

    # 3) vLLM smoke test는 CUDA+vLLM 있을 때만
    try:
        import torch
        if not torch.cuda.is_available():
            print("[SKIP] CUDA 없음 → vLLM 테스트는 생략 (제출은 가능)")
            return
    except Exception:
        print("[SKIP] torch 확인 실패 → vLLM 테스트 생략")
        return

    try:
        from vllm import LLM, SamplingParams
    except Exception:
        print("[SKIP] vLLM 미설치 → vLLM 테스트 생략")
        return

    messages = [{"role": "user", "content": "한국의 수도는?"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=str(model_dir), trust_remote_code=args.trust_remote_code, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    out = llm.generate([prompt], sp)[0].outputs[0].text
    print("[OK] vLLM 추론 성공")
    print(out.strip())


if __name__ == "__main__":
    main()
