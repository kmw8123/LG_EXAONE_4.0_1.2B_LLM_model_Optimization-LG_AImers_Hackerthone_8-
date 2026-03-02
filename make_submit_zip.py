# -*- coding: utf-8 -*-
import argparse
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
EXCLUDE_FILES = {".DS_Store", "Thumbs.db"}

REQUIRED_FILES = [
    "config.json",
    "model.safetensors",  # GPTQ/일반 safetensors 모두 보통 이 이름을 씀
    "tokenizer.json",
    "tokenizer_config.json",
]

OPTIONAL_BUT_RECOMMENDED = [
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "recipe.yaml",
]


def iter_model_files(model_dir: Path):
    for p in model_dir.rglob("*"):
        if p.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name in EXCLUDE_FILES:
            continue
        yield p


def main():
    parser = argparse.ArgumentParser(
        prog="make_submit_zip.py",
        description="Create submit.zip with top-level 'model/' directory for Phase2 submission."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="제출용 모델 파일들이 들어있는 폴더 (예: C:\\Users\\...\\submit_model_export)"
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default="submit.zip",
        help="생성할 submit.zip 경로 (기본: ./submit.zip)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["stored", "deflated"],
        default="deflated",
        help="zip 압축 방식 (stored=무압축, deflated=압축). 기본 deflated"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    zip_path = Path(args.zip_path).resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"--model_dir 경로가 폴더가 아닙니다: {model_dir}")

    # 필수 파일 체크
    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise RuntimeError(
            "제출용 model_dir에 필수 파일이 없습니다:\n"
            + "\n".join([f"- {x}" for x in missing])
            + f"\n\n현재 model_dir: {model_dir}"
        )

    # 권장 파일 경고(실패는 아님)
    warn_missing = [f for f in OPTIONAL_BUT_RECOMMENDED if not (model_dir / f).exists()]
    if warn_missing:
        print("[WARN] 아래 파일이 없지만 제출은 진행합니다. (권장 파일)")
        for f in warn_missing:
            print(f"  - {f}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    compression = zipfile.ZIP_DEFLATED if args.compression == "deflated" else zipfile.ZIP_STORED

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
        for fpath in iter_model_files(model_dir):
            rel = fpath.relative_to(model_dir).as_posix()
            arcname = f"model/{rel}"  # ★ 최상위 model/ 고정
            zf.write(fpath, arcname)
            file_count += 1

    # zip 내부 구조 검증
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        has_model_prefix = any(n.startswith("model/") for n in names)
        bad_top = [n for n in names if not n.startswith("model/")]

    if not has_model_prefix:
        raise RuntimeError("[FAIL] zip 내부에 'model/' 폴더가 없습니다. 생성 로직을 확인하세요.")

    if bad_top:
        # 거의 발생하지 않지만, 안전장치
        print("[WARN] zip 최상위에 model/ 이외 항목이 있습니다(권장: model/만 존재).")
        for n in bad_top[:20]:
            print("  -", n)
        if len(bad_top) > 20:
            print(f"  ... ({len(bad_top)} items)")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] submit.zip 생성 완료: {zip_path} ({size_mb:.1f} MB, files={file_count})")
    print("[CHECK] zip을 열었을 때 최상위에 'model/' 폴더가 보이면 정상입니다.")


if __name__ == "__main__":
    main()