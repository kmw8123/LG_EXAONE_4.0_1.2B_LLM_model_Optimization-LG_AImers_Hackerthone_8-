# -*- coding: utf-8 -*-
"""
train_lora_sft.py
- EXAONE-4.0-1.2B 기반 LoRA SFT (MANTA-1M 같은 대화 데이터)
- chat_template.jinja 기반으로 텍스트 구성
- loss는 "assistant turn"에만 걸도록 마스킹(가능한 한 안정적인 성능 향상 목적)

주의:
- QLoRA(4bit) 학습을 하려면 bitsandbytes가 필요할 수 있음.
- Windows 환경에서 bnb가 어려우면, Colab/Kaggle/WSL2에서 학습 후 결과(adapter)만 가져오는 걸 추천.

학습 결과(out_dir) = LoRA adapter 폴더
그 다음 export_submit_model.py의 --lora_dir로 merge 가능.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as e:
    raise RuntimeError("peft가 필요합니다. pip install peft") from e


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_conv_field(ex: Dict[str, Any]) -> Optional[Any]:
    for k in ["conversations", "messages", "chat", "dialog", "dialogue"]:
        if k in ex and ex[k] is not None:
            return ex[k]
    return None


def build_text(tokenizer, ex: Dict[str, Any]) -> Optional[str]:
    conv = pick_conv_field(ex)
    if conv is None:
        return None
    try:
        # 학습용이므로 generation prompt는 False (assistant 답변까지 포함된 전체 대화)
        return tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    except Exception:
        return None


def find_target_module_suffixes(model) -> List[str]:
    """
    PEFT의 target_modules는 보통 suffix 매칭이라,
    Linear layer 이름의 "끝부분"들을 모아서 전달하는 방식이 안정적.
    """
    suffixes = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(skip in name for skip in ["lm_head", "embed_tokens"]):
                continue
            suffixes.add(name.split(".")[-1])
    # 흔히 나오는 것 우선 정렬
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    out = [x for x in preferred if x in suffixes]
    out += sorted([x for x in suffixes if x not in set(out)])
    return out


def build_labels_only_on_assistant(
    tokenizer,
    input_ids: List[int],
    assistant_prefix: str = "[|assistant|]\n",
    end_of_turn: str = "[|endofturn|]\n",
) -> Optional[List[int]]:
    """
    chat_template에서 assistant 구간만 loss를 주도록 라벨 마스킹.
    - assistant_prefix 이후 ~ 다음 end_of_turn 전까지 라벨=정답
    - 나머지는 -100
    """
    pref = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    eot = tokenizer.encode(end_of_turn, add_special_tokens=False)
    if not pref or not eot:
        # 특수 토큰이 아닌 일반 문자열로 처리되는 경우도 있어 length=0일 수 있음
        # 그땐 안전하게 전체 loss로 fallback
        return input_ids[:]

    labels = [-100] * len(input_ids)

    def find_seq(hay: List[int], needle: List[int], start: int) -> int:
        n = len(needle)
        for i in range(start, len(hay) - n + 1):
            if hay[i : i + n] == needle:
                return i
        return -1

    pos = 0
    found_any = False
    while True:
        a = find_seq(input_ids, pref, pos)
        if a < 0:
            break
        start = a + len(pref)
        b = find_seq(input_ids, eot, start)
        end = b if b >= 0 else len(input_ids)
        if start < end:
            for i in range(start, end):
                labels[i] = input_ids[i]
            found_any = True
        pos = end + len(eot)

    if not found_any:
        # assistant turn 탐지 실패 시 전체 loss
        return input_ids[:]
    return labels


class SimpleCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features: input_ids, attention_mask, labels
        # pad
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            if max_len % m != 0:
                max_len = ((max_len // m) + 1) * m

        batch_input_ids = []
        batch_attn = []
        batch_labels = []

        pad_id = self.tokenizer.pad_token_id
        for f in features:
            ids = f["input_ids"]
            attn = f["attention_mask"]
            lab = f["labels"]

            pad_n = max_len - len(ids)
            batch_input_ids.append(ids + [pad_id] * pad_n)
            batch_attn.append(attn + [0] * pad_n)
            batch_labels.append(lab + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--out_lora_dir", type=str, required=True)
    parser.add_argument("--chat_template_path", type=str, required=True)

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")

    # LoRA 하이퍼 (보수적으로 시작 -> 성능 더 필요하면 r 올리기)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 학습 하이퍼
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # 4bit(QLO) 옵션 (가능하면 켜는 걸 추천)
    parser.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_lora_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer + chat_template
    chat_template = read_text(args.chat_template_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=args.trust_remote_code)
    tokenizer.chat_template = chat_template

    if tokenizer.pad_token_id is None:
        # 보통 pad가 없으면 eos로 설정
        tokenizer.pad_token = tokenizer.eos_token

    # model load (4bit 가능하면 사용)
    quantization_config = None
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("[INFO] QLoRA(4bit) 로딩 사용")
        except Exception as e:
            print("[WARN] BitsAndBytesConfig 사용 불가 -> use_4bit=False로 fallback 권장")
            print(f"[WARN] 원인: {e}")
            quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    # target modules 자동 추출
    target_suffixes = find_target_module_suffixes(model)
    print("[INFO] LoRA target suffixes:", target_suffixes)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_suffixes,
    )
    model = get_peft_model(model, lora_cfg)

    # dataset
    print("[INFO] dataset load...")
    ds = load_dataset(args.dataset_id, split=args.dataset_split)
    ds = ds.shuffle(seed=args.seed)

    if args.train_samples < len(ds):
        ds = ds.select(range(args.train_samples))

    # tokenize + labels(assistant only)
    def preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        txt = build_text(tokenizer, ex)
        if not txt:
            return {}

        enc = tokenizer(
            txt,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        if len(input_ids) < 64:
            return {}

        labels = build_labels_only_on_assistant(tokenizer, input_ids)
        attn = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x.get("input_ids") is not None and len(x["input_ids"]) > 0)

    print("[INFO] train samples(after filter):", len(ds))

    collator = SimpleCollator(tokenizer)

    # training args
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit" if quantization_config is not None else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("[INFO] saving LoRA adapter...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # chat_template도 같이 저장
    with open(out_dir / "chat_template.jinja", "w", encoding="utf-8") as f:
        f.write(chat_template)

    print(f"[OK] LoRA 저장 완료: {out_dir}")


if __name__ == "__main__":
    main()
