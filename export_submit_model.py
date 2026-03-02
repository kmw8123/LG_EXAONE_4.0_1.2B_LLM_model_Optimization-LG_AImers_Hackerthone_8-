import argparse
import shutil
from pathlib import Path
from typing import Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def has_any_tokenizer_files(d: Path) -> bool:
    cand = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    ]
    return any((d / x).exists() for x in cand)


def has_any_weight_files(d: Path) -> bool:
    return any(d.glob("*.safetensors")) or any(d.glob("*.bin"))


def copy_filtered_files(
    src_dir: Path,
    dst_dir: Path,
    *,
    exclude_names: Optional[Set[str]] = None,
    exclude_suffixes: Optional[Set[str]] = None,
) -> None:
    exclude_names = exclude_names or set()
    exclude_suffixes = exclude_suffixes or set()

    for p in src_dir.iterdir():
        if p.is_dir():
            continue
        name = p.name
        suffix = p.suffix.lower()

        if name in exclude_names:
            continue
        if suffix in exclude_suffixes:
            continue
        if name.startswith("."):
            continue

        shutil.copy2(p, dst_dir / name)


def ensure_generation_config(out_dir: Path, model) -> None:
    gen_path = out_dir / "generation_config.json"
    if gen_path.exists():
        return
    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is None:
            return
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(gen_cfg.to_json_string())
    except Exception:
        return


def main():
    parser = argparse.ArgumentParser(description="Create submit-ready HF model folder (model/).")
    parser.add_argument("--out_dir", type=str, default="model")
    parser.add_argument("--chat_template_path", type=str, default="chat_template.jinja")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--src_model_dir", type=str, default=None, help="이미 완성된 HF 모델 디렉토리(복사 모드)")
    group.add_argument("--lora_dir", type=str, default=None, help="PEFT LoRA adapter 디렉토리(merge 모드)")

    parser.add_argument("--base_model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument(
        "--tokenizer_src",
        type=str,
        default=None,
        help="토크나이저 로드 소스(로컬 폴더 또는 HF repo). 지정 안 하면 src_model_dir에 tokenizer가 있으면 거기, 없으면 base_model_id 사용",
    )

    parser.add_argument("--dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16"))
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recipe_path", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chat_template = read_text(args.chat_template_path)

    # 0) src_model_dir면 먼저 존재/유효성 검사 (← 여기 때문에 네 repo id 에러를 원천 차단)
    src_dir = None
    if args.src_model_dir:
        src_dir = Path(args.src_model_dir).resolve()
        if not src_dir.exists():
            raise FileNotFoundError(
                f"--src_model_dir 폴더가 없습니다: {src_dir}\n"
                f"보통 GPTQ 단계가 실패해서 out_dir가 생성되지 않은 경우입니다."
            )
        if not (src_dir / "config.json").exists():
            raise RuntimeError(
                f"--src_model_dir에 config.json이 없습니다: {src_dir}\n"
                f"폴더가 비어있거나 모델 저장이 실패했을 가능성이 큽니다."
            )
        if not has_any_weight_files(src_dir):
            raise RuntimeError(
                f"--src_model_dir에 weight 파일(*.safetensors or *.bin)이 없습니다: {src_dir}\n"
                f"폴더가 비어있거나 모델 저장이 실패했을 가능성이 큽니다."
            )

    # 1) Tokenizer 소스 결정
    if args.tokenizer_src:
        tok_src = args.tokenizer_src
    else:
        if src_dir and has_any_tokenizer_files(src_dir):
            tok_src = str(src_dir)
        else:
            tok_src = args.base_model_id

    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(out_dir)
    shutil.copy2(Path(args.chat_template_path), out_dir / "chat_template.jinja")

    # 2) Model
    if args.src_model_dir:
        # copy-only (양자화/압축 포맷 보호)
        exclude_names = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "tokenizer.model",
            "spiece.model",
            "chat_template.jinja",
            "README.md",
            "README.txt",
            "LICENSE",
            "LICENSE.txt",
            ".gitattributes",
        }
        exclude_suffixes = {".py", ".ipynb", ".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".log"}
        copy_filtered_files(src_dir, out_dir, exclude_names=exclude_names, exclude_suffixes=exclude_suffixes)

        # recipe.yaml 포함
        if args.recipe_path:
            shutil.copy2(Path(args.recipe_path), out_dir / "recipe.yaml")
        elif (src_dir / "recipe.yaml").exists():
            shutil.copy2(src_dir / "recipe.yaml", out_dir / "recipe.yaml")

    else:
        torch_dtype = None
        if args.dtype == "float16":
            torch_dtype = torch.float16
        elif args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_dir)
        model = model.merge_and_unload()
        model.save_pretrained(out_dir, safe_serialization=True)
        ensure_generation_config(out_dir, model)

        if args.recipe_path:
            shutil.copy2(Path(args.recipe_path), out_dir / "recipe.yaml")

    # 3) 최종 검증(빈 폴더 방지)
    if not (out_dir / "config.json").exists():
        raise RuntimeError("out_dir에 config.json이 없습니다. export 실패")
    if not has_any_weight_files(out_dir):
        raise RuntimeError("out_dir에 weight 파일이 없습니다. export 실패")

    print(f"[OK] 제출용 model/ 폴더 생성 완료: {out_dir}")
    print("[NEXT] make_submit_zip.py로 submit.zip 생성")


if __name__ == "__main__":
    main()