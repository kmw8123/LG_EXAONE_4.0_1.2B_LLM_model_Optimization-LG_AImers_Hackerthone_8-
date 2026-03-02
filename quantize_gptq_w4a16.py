import argparse
import random
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description="GPTQ(W4A16) quantization with llmcompressor for Phase2 제출용 모델.")
    parser.add_argument("--model_dir", type=str, required=True, help="베이스 모델 로컬 경로 (예: ./base_model)")
    parser.add_argument("--out_dir", type=str, required=True, help="양자화된 모델 저장 폴더")
    parser.add_argument("--chat_template_path", type=str, default="chat_template.jinja")

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--num_calibration_samples", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU가 필요합니다. (GPTQ는 GPU에서 수행하는 것을 강력 권장)")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model_dir = args.model_dir
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Tokenizer (+ chat_template 주입)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    tokenizer.chat_template = read_text(args.chat_template_path)

    # 2) Calibration dataset
    #    "앞부분만 쓰면 편향될 수 있어서" pool을 조금 넓게 잡고 shuffle 후 앞 N개 사용
    pool = max(args.num_calibration_samples * 4, args.num_calibration_samples)
    ds = load_dataset(args.dataset_id, split=f"{args.dataset_split}[:{pool}]")
    if pool > args.num_calibration_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.num_calibration_samples))

    def preprocess(example):
        conv = example.get("conversations")
        if conv is None:
            # fallback: 일반 텍스트 데이터셋일 때
            text = example.get("text") or example.get("content") or ""
            conv = [{"role": "user", "content": text}]
        return {"text": tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)}

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    # 3) Load model
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("cuda")

    # 4) GPTQ with llmcompressor
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except Exception as e:
        raise SystemExit(
            "llmcompressor import 실패. 아래 패키지를 설치하세요:\n"
            "  pip install llmcompressor datasets transformers\n"
            f"원인: {e}"
        )

    recipe = [
        GPTQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=["embed_tokens", "lm_head"],
        )
    ]

    print(f"[INFO] GPTQ 시작: samples={args.num_calibration_samples}, max_seq_len={args.max_seq_len}")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calibration_samples,
    )
    print("[INFO] GPTQ 완료")

    # 5) Save compressed model
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    shutil.copy2(Path(args.chat_template_path), out_dir / "chat_template.jinja")

    # 기록용 recipe.yaml (제출/리뷰용)
    recipe_yaml = f"""# Quantization recipe (record)
quantization:
  method: GPTQ
  scheme: W4A16
  targets: [Linear]
  ignore: [embed_tokens, lm_head]
calibration:
  dataset_id: {args.dataset_id}
  dataset_split: {args.dataset_split}
  num_calibration_samples: {args.num_calibration_samples}
  max_seq_len: {args.max_seq_len}
seed: {args.seed}
"""
    write_text(out_dir / "recipe.yaml", recipe_yaml)

    print(f"[OK] 양자화 모델 저장 완료: {out_dir}")
    print("[NEXT] export_submit_model.py --src_model_dir <out_dir> 로 제출용 model 폴더를 만들고 zip 하세요.")


if __name__ == "__main__":
    main()