#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Dict, Any, Tuple
import datetime
import torch
import torch.multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from PIL import Image
from queue import Empty
from contextlib import nullcontext
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

import threading
import time
import sys
from datetime import datetime
import re

from packages.file_deal import read_dataset, save_data_result
from packages.constants import (
    PROJECT_ROOT,
    EVAL_RES_OUTPUT_PATH,
    EVAL_DATA_IMAGE_ROOT,
    EVAL_DATA_EXCEL_PATH,
    LLM_SCORE_OUTPUT_PATH,
    API_URL,
    API_KEY_DISCOUNT,
    GPT_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    CLAUDE_MODEL_NAME,
    GEMINI_MODEL_NAME,
)

# 仅移除生成文本中的图片占位符，不影响 <think> 内容
_IMG_TAG = re.compile(r"(?:<\|image\|>)+", re.S)
# 捕获思考段（用于单独保存思考内容）
_THINK_CAP = re.compile(r"<think>(.*?)</think>", re.S)
# 捕获答案部分
_BOX_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.S)

# import re

# # === 展示层清洗：去掉 <|image|>、<think>...，优先提取盒子答案 ===
# _IMG_TAG   = re.compile(r"(?:<\|image\|>)+", re.S)
# _THINK_TAG = re.compile(r"<think>.*?</think>", re.S)
# _BOX_RE    = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.S)

# def extract_clean_answer(text: str) -> str:
#     # 优先取盒子答案
#     m = _BOX_RE.search(text)
#     if m:
#         return m.group(1).strip()
#     # 去掉思考草稿与图片占位
#     text = _THINK_TAG.sub("", text)
#     text = _IMG_TAG.sub("", text)
#     # 防守式清理其它 <|...|> 标记（如你的模板需要保留，请注释掉这行）
#     text = re.sub(r"<\|.*?\|>", "", text)
#     return text.strip()


# =========================================================
# 工具函数
# =========================================================
class ColoredLogger:
    """颜色控制类"""

    # 颜色代码
    COLORS = {
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "MAGENTA": "\033[95m",
        "CYAN": "\033[96m",
        "WHITE": "\033[97m",
        "RESET": "\033[0m",
    }

    # 日志级别颜色映射
    LEVEL_COLORS = {
        "INFO": "GREEN",
        "DEBUG": "BLUE",
        "WARNING": "YELLOW",
        "ERROR": "RED",
        "CRITICAL": "MAGENTA",
    }

    @classmethod
    def colorize(cls, text, color_name):
        """为文本添加颜色"""
        color_code = cls.COLORS.get(color_name.upper(), cls.COLORS["RESET"])
        return f"{color_code}{text}{cls.COLORS['RESET']}"

    @classmethod
    def colorize_by_level(cls, text, level):
        """根据日志级别为文本添加颜色"""
        color_name = cls.LEVEL_COLORS.get(level, "WHITE")
        return cls.colorize(text, color_name)


class IntervalLogger(threading.Thread):
    """
    固定间隔打印日志的多线程类

    参数:
        name: 日志器名称，用于前缀区分
        interval: 打印间隔(秒)
        color: 日志颜色(可选: RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE)
        level: 日志级别(可选: INFO, DEBUG, WARNING, ERROR, CRITICAL)
    """

    def __init__(self, name, target_gpus, interval=2.0, color=None, level="INFO"):
        super().__init__()
        self.name = name
        self.interval = interval
        self.color = color
        self.level = level
        self._stop_event = threading.Event()
        self.daemon = True  # 设置为守护线程，主程序退出时自动结束
        self.target_gpus = target_gpus
        self.last_mem_alloc_list = [0 for _ in target_gpus]
        self.last_mem_resv_list = [0 for _ in target_gpus]
        self.current_mem_alloc_list = [0 for _ in target_gpus]
        self.current_mem_resv_list = [0 for _ in target_gpus]
        self.count = 0

    def run(self):
        """线程主函数"""
        while not self._stop_event.is_set():
            self.log()
            time.sleep(self.interval)
            self.count += 1

    def log(self):
        """格式化并打印日志消息"""
        message = "GPU内存使用情况:"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建前缀
        prefix = f"[{timestamp}][{self.name}][{self.level}]"

        # 添加颜色
        if self.color:
            colored_prefix = ColoredLogger.colorize(prefix, self.color)
            formatted_message = f"{colored_prefix} {message}"
        else:
            # 如果没有指定颜色，使用级别对应的颜色
            colored_prefix = ColoredLogger.colorize_by_level(prefix, self.level)
            formatted_message = f"{colored_prefix} {message}"

        # 打印消息
        msg_to_print_list = [formatted_message]
        sys.stdout.flush()  # 确保立即输出
        # 默认不打印，符合条件才打印，减少无效日志
        is_print = False
        # 挨个检查每个GPU
        for logical_idx, physical_id in enumerate(self.target_gpus):
            try:
                mem_alloc = (
                    torch.cuda.memory_allocated(logical_idx) / 1024 / 1024 / 1024
                )
                self.current_mem_alloc_list[logical_idx] = mem_alloc
                mem_resv = torch.cuda.memory_reserved(logical_idx) / 1024 / 1024 / 1024
                self.current_mem_resv_list[logical_idx] = mem_resv
                msg_to_print_list.append(
                    f"GPU {physical_id}: 已分配 {mem_alloc:.2f}GB(delta={mem_alloc-self.last_mem_alloc_list[logical_idx]:.2f}), 已保留 {mem_resv:.2f}GB(delta={mem_resv-self.last_mem_resv_list[logical_idx]:.2f})"
                )
                # 间隔大于一分钟强行输出一次日志；内存增长超过5GB时打印一次日志
                if (self.count * self.interval % 60 == 0) or (
                    mem_alloc > self.last_mem_alloc_list[logical_idx] + 5
                    or mem_resv > self.last_mem_resv_list[logical_idx] + 5
                ):
                    # 有一个GPU符合条件就全部打印一次
                    is_print = True

            except Exception as e:
                print(e)
                msg_to_print_list.append(f"无法获取GPU {physical_id} 的内存使用情况")

        if is_print:
            for msg in msg_to_print_list:
                print(msg)
            # 存下本次打印的数据
            self.last_mem_alloc_list = self.current_mem_alloc_list[:]
            self.last_mem_resv_list = self.current_mem_resv_list[:]

    def stop(self):
        """停止日志打印"""
        self._stop_event.set()


def select_devices_and_max_memory(tp_size: int, util: float):
    """
    只用前 tp_size 张可见 GPU；把每张卡的可用显存预算成总显存 * util * 0.98（留 2% buffer）。
    返回 device_map 与 max_memory；在 CPU 情况下返回 (None, None)。
    """
    assert 0 < util <= 1.0
    if not torch.cuda.is_available():
        return None, None

    num_visible = torch.cuda.device_count()
    use_n = max(1, min(tp_size, num_visible))
    choose = list(range(use_n))  # 逻辑编号 0..use_n-1（受 CUDA_VISIBLE_DEVICES 影响）

    max_memory = {}
    for i in choose:
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        budget_gib = max(1.0, total_gib * util * 0.98)
        # 关键点：键必须是整数，而不是 "cuda:0"
        max_memory[i] = f"{int(budget_gib)}GiB"

    return "auto", max_memory


def load_image(path: str, max_long: int = 672) -> Image.Image:
    """
    读取并按最长边 max_long 等比缩放图像（减少视觉 token 数）。
    """
    img = Image.open(path).convert("RGB")
    if max_long and max(img.size) > max_long:
        img.thumbnail((max_long, max_long), Image.Resampling.LANCZOS)
    return img


def load_image_with_pixel_budget(
    path: str, min_pixels: int, max_pixels: int
) -> Image.Image:
    """
    读取图像，并将像素总数限制在 [min_pixels, max_pixels] 区间。
    仅在像素过大时按比例缩小；像素不足时不放大（避免无益的插值）。
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    pix = w * h
    if max_pixels and pix > max_pixels:
        scale = (max_pixels / pix) ** 0.5
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # 若你确实想强制不低于 min_pixels，可在此处按需上采样（一般不建议）
    return img


def process_batch_on_gpu(
    rank: int, input_queue: mp.Queue, output_queue: mp.Queue, model_path: str, args: Any
):
    """
    GPU worker进程函数，负责处理单个batch的推理

    Args:
        rank: GPU编号
        input_queue: 输入队列，包含待处理的batch数据
        output_queue: 输出队列，用于返回结果
        model_path: 模型路径
        args: 其他参数
    """
    # Worker processes should not load models - they should receive them
    # But since we're using multiprocessing, we need to redesign this
    
    try:
        # Simply process batches as they come in - no model loading
        while True:
            try:
                # 从队列获取数据
                batch_data = input_queue.get(timeout=5)
                if batch_data is None:  # 退出信号
                    break

                # This approach won't work with current design because model can't be pickled
                # A better approach is to restructure to avoid loading model in each worker
                print(f"GPU {rank} received batch but cannot process without model")
                output_queue.put((None, None))

            except Empty:
                continue
            except Exception as e:
                print(f"GPU {rank} batch processing error: {e}")
                output_queue.put((None, None))
                break

    except Exception as e:
        print(f"GPU {rank} worker failed: {e}")
        output_queue.put((None, None))


def build_messages(image: Image.Image, query_text: str) -> List[Dict[str, Any]]:
    """
    GLM-4.5V 多模态对话消息格式。和你原逻辑一致。
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query_text},
            ],
        },
    ]


def apply_chat_and_pack(
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
) -> Tuple[str, Image.Image]:
    """
    用 processor.apply_chat_template 生成文本 prompt，并返回 (prompt, image)。
    注意：我们把图像对象留给 processor(images=...) 来做视觉前处理。
    """
    # 取出 image（第一个 user content 中的 image）
    pil_image = None
    for seg in messages:
        if seg["role"] == "user":
            for c in seg["content"]:
                if c.get("type") == "image":
                    pil_image = c["image"]
                    break
    assert pil_image is not None, "No image found in messages."

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},  # ← 关闭思考
    )
    return prompt, pil_image


@torch.inference_mode()
def generate_batch(
    model,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    images: List[Image.Image],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    attn_implementation: str = None,
    args=None,  # 需要传入
) -> List[str]:
    # 1) 打包输入；文本侧按 max_model_len 截断，避免构造时已溢出
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=(args.max_model_len if args is not None else 16384),
    )

    # 2) 多卡分片：保持在 CPU 由 Accelerate 搬运；单卡再搬到该卡
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map is None and torch.cuda.is_available():
        dev = next(model.parameters()).device
        inputs = {
            k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }

    # 3) 逐条样本计算“前缀长度”，得到每条的最大可生成步数；取全 batch 的最小值
    attn = inputs.get("attention_mask", None)
    # 同时用 pad 掩码与 attention_mask 估计前缀长度，取较大值，避免低估
    pad_mask = None
    if getattr(tokenizer, "pad_token_id", None) is not None and "input_ids" in inputs:
        pad_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()
    if attn is not None and pad_mask is not None:
        in_lens = torch.maximum(attn.sum(dim=1), pad_mask.sum(dim=1))
    elif attn is not None:
        in_lens = attn.sum(dim=1)
    elif pad_mask is not None:
        in_lens = pad_mask.sum(dim=1)
    else:
        # 没有 mask 时，用序列列数兜底
        in_lens = torch.tensor(
            [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
        )

    # 每条的可用预算
    per_sample_cap = (args.max_model_len - in_lens - 1).clamp(min=1)
    # 取本 batch 能统一使用的 cap
    batch_cap = int(per_sample_cap.min().item())
    eff_max_new = max(1, min(max_new_tokens, batch_cap))
    print(
        f"[DBG] prefix_lens[:4]={in_lens[:4].tolist()}, eff_max_new={eff_max_new}, "
        f"want={max_new_tokens}, max_len={args.max_model_len}"
    )

    # 4) 生成参数
    gen_kwargs = dict(
        max_new_tokens=eff_max_new,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        use_cache=True,
        return_dict_in_generate=True,  # 方便拿到 sequences
    )

    # 5) 避免不被模型使用的键
    generate_inputs = dict(inputs)
    generate_inputs.pop("token_type_ids", None)

    # 6) 生成
    out = model.generate(**generate_inputs, **gen_kwargs)
    seqs = out.sequences if hasattr(out, "sequences") else out  # [B, L_total]

    # 7) 只解码“新增部分”，避免把提示一起解码看起来像被截断
    prefix_lens = in_lens.tolist()
    texts = []
    for i, seq in enumerate(seqs):
        gen_ids = seq[prefix_lens[i] :]  # 取新增 token
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        txt = _IMG_TAG.sub("", txt)
        texts.append(txt.strip())

    # 8) 如果被预算压缩过，打个调试日志
    if eff_max_new < max_new_tokens:
        print(
            f"[WARN] max_new_tokens 被压到 {eff_max_new}（受 max_model_len={args.max_model_len} 与前缀长度约束）"
        )

    return texts


def exact_match(pred: str, gt: str) -> bool:
    """
    最简单精确匹配，你可以按需替换为更鲁棒的比较（去标点/空格/大小写等）。
    """
    return pred.strip() == gt.strip()


def extract_final_cn(s: str, *, strip_edge_punct: bool = False) -> str:

    # # 中文汉字 + 常见中文标点（含全角），另补 U+00B7 ·、U+2014 —、U+2026 …
    # _CN_BLOCKS = (
    #     r"\u3400-\u4DBF"  # CJK Ext A
    #     r"\u4E00-\u9FFF"  # CJK Unified Ideographs
    #     r"\uF900-\uFAFF"  # CJK Compatibility Ideographs
    #     r"\u3000-\u303F"  # CJK Symbols & Punctuation（。、“”、——、…… 等）
    #     r"\uFF01-\uFF0F"  # 全角标点片段
    #     r"\uFF1A-\uFF20"
    #     r"\uFF3B-\uFF40"
    #     r"\uFF5B-\uFF65"
    #     r"\u00B7"  # ·
    #     r"\u2014"  # —
    #     r"\u2026"  # …
    # )

    # # 1) 抓“最后一对花括号”里的中文：   {...中文...}，要求此处的 '}' 之后**不再出现 '{'**
    # #    这样能拿到 \text{中文} 这种**内层**的中文（通常就是我们要的）。
    # _BRACED_LAST_CN = re.compile(r"\{([" + _CN_BLOCKS + r"]+)\}(?!.*\{)", re.S)

    # # 2) 兜底：整串里“最后一段中文”
    # _LAST_CN_SPAN = re.compile(r"[" + _CN_BLOCKS + r"]+")

    # s = (s or "").strip()
    # m = _BRACED_LAST_CN.search(s)
    # if m:
    #     out = m.group(1)
    # else:
    #     ms = list(_LAST_CN_SPAN.finditer(s))
    #     out = ms[-1].group(0) if ms else ""

    # if strip_edge_punct and out:
    #     # 如需去掉首尾中文标点（保留中间的），打开这个选项
    #     edge_punct = re.compile(
    #         r"^[" + _CN_BLOCKS + r"&&\p{P}]*|[" + _CN_BLOCKS + r"&&\p{P}]*$"
    #     )
    #     try:
    #         import regex  # 可选：如果安装了 regex，可以更精准判断标点

    #         out = regex.sub(r"^\p{P}+", "", out)
    #         out = regex.sub(r"\p{P}+$", "", out)
    #     except Exception:
    #         # 退化：常见尾句号/逗号等
    #         out = re.sub(r"^[，。、“”‘’！？：；《》【】…—·]+", "", out)
    #         out = re.sub(r"[，。、“”‘’！？：；《》【】…—·]+$", "", out)
    out = s
    if out.startswith('这道菜叫') or out.startswith('这道菜是'):
        out = out[4:]
    out = out.strip('。“”*')
    return out


# =========================================================
# 主评估流程（Torch 版本）
# =========================================================


def evaluate_model_accuracy(
    model_path: str,
    samples_df: pd.DataFrame,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    model,
    output_dir: str,
    img_root: str,
    image_max_long: int = 672,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    args=None,
):
    """评估模型准确率的主函数，使用已加载的模型进行推理"""
    
    os.makedirs(output_dir, exist_ok=True)

    results = []
    correct, total = 0, 0

    print(f"Preparing inputs for model evaluation...")

    packed = []  # [(prompt, pil_image, meta_dict), ...]
    for idx, row in tqdm(
        list(enumerate(samples_df.itertuples(index=False))),
        total=len(samples_df),
        desc="Preparing prompts",
    ):
        try:
            orig_path = row.image_path
            # 将 Excel 里存的相对/原路径替换为统一根目录
            img_path = os.path.join(img_root, os.path.basename(orig_path))
            if not os.path.exists(img_path):
                print(
                    f"[WARN] image missing: {img_path} (index={idx}, orig={orig_path})"
                )
                continue
            img = load_image_with_pixel_budget(
                img_path, min_pixels=args.mm_min_pixels, max_pixels=args.mm_max_pixels
            )
            question = "这道菜叫什么名字？ 请直接提供答案"
            gt = str(row.food_name)
            ttype = str(row.type)

            messages = build_messages(img, question)
            prompt, pil_img = apply_chat_and_pack(processor, messages)

            packed.append(
                (
                    prompt,
                    pil_img,
                    dict(
                        sample_index=idx,
                        original_image_path=orig_path,
                        modified_image_path=img_path,
                        problem=question,
                        ground_truth_answer=gt,
                        type=ttype,
                    ),
                )
            )
        except Exception as e:
            print(
                f"[ERR] prepare index={idx}, path={getattr(row, 'image_path', 'N/A')}: {e}"
            )

    if not packed:
        print("No valid samples to process. Exit.")
        return

    print(f"\nGenerating responses for {len(packed)} samples...")

    # Process in batches using the already loaded model
    for i in tqdm(range(0, len(packed), batch_size), desc="Processing batches"):
        chunk = packed[i : i + batch_size]
        prompts = [c[0] for c in chunk]
        images = [c[1] for c in chunk]
        metas = [c[2] for c in chunk]

        # Execute inference with the already loaded model
        with torch.inference_mode():
            texts = generate_batch(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                prompts=prompts,
                images=images,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(args.temperature > 0.0),
                args=args,
            )

        # Process results
        for meta, pred in zip(metas, texts):
            try:
                # 单独提取思考段，保留在结果中；不影响 pred 本身
                _m = _THINK_CAP.search(pred)
                model_thinking = _m.group(1).strip() if _m else ""
                model_answer = (
                    extract_final_cn(pred) if extract_final_cn(pred) else pred
                )
                is_ok = exact_match(model_answer, meta["ground_truth_answer"])
                correct += int(is_ok)
                total += 1
                results.append(
                    {
                        **meta,
                        "model_prediction": model_answer,
                        "model_thinking": model_thinking,
                        "is_correct": is_ok,
                        "raw_answer": pred,
                    }
                )
            except Exception as e:
                print(f"Error processing result: {e}")

    # Calculate final accuracy and save results
    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"\nOverall Accuracy: {acc:.2f}% ({correct}/{total})")

    # Save results
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = os.path.basename(model_path.rstrip("/"))
    out_file = os.path.join(
        output_dir, f"{current_time}_evaluation_results_{suffix}_by_local.csv"
    )
    save_data_result(results, out_file)
    print(f"Saved {total} results to {out_file}")


# =========================================================
# 入口：加载模型（Torch / HF），读表并评估
# =========================================================


def main():
    ap = argparse.ArgumentParser("Evaluate model path.")
    ap.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=True,
        help="本地或 HF 上的 GLM-4.5V 模型目录",
    )
    ap.add_argument(
        "-o", "--output_path", type=str, default=EVAL_RES_OUTPUT_PATH, help="输出目录"
    )
    ap.add_argument(
        "--input_excel_path",
        type=str,
        default=EVAL_DATA_EXCEL_PATH,
        help="包含 image_path / food_name / type 列的 Excel 路径",
    )
    ap.add_argument(
        "--input_img_path",
        type=str,
        default=EVAL_DATA_IMAGE_ROOT,
        help="图片所在目录（会用 basename 拼接）",
    )
    ap.add_argument(
        "--max_test_img_num",
        type=int,
        default=0,
        help="最大测试数量,用于调试,0代表不控制",
    )

    # 生成相关
    # 调试可以用10个加快测试速度
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)

    # 图像与长度控制
    ap.add_argument(
        "--image_max_long",
        type=int,
        default=0,
        help="最长边缩放（像素），减少视觉 token；0 表示不缩放",
    )

    ap.add_argument(
        "--max_model_len",
        type=int,
        default=4096 * 4,
        help="等价于 vLLM 的 max_model_len",
    )
    ap.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="等价于 vLLM 的 tensor_parallel_size（选用前 N 张 GPU）",
    )
    ap.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="等价于 vLLM 的 gpu_memory_utilization，用于构造 max_memory",
    )
    ap.add_argument(
        "--mm_min_pixels",
        type=int,
        default=256 * 28 * 28,
        help="等价于 vLLM 的 mm_processor_kwargs.min_pixels",
    )
    ap.add_argument(
        "--mm_max_pixels",
        type=int,
        default=1280 * 28 * 28,
        help="等价于 vLLM 的 mm_processor_kwargs.max_pixels",
    )

    # 模型加载控制
    ap.add_argument(
        "--bf16", action="store_true", default=True, help="使用 bfloat16（H100 推荐）"
    )
    ap.add_argument(
        "--attn_impl",
        type=str,
        default=None,
        choices=[None, "flash_attention_2", "sdpa", "eager"],
        help="部分模型支持指定注意力实现（可留空由模型自行选择）",
    )

    args = ap.parse_args()

    # 读取 Excel
    print(f"Loading dataset from Excel: {args.input_excel_path} ...")
    try:
        df = read_dataset(args.input_excel_path)
        required = ["image_path", "food_name", "type"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            print(f"[ERR] Missing columns: {', '.join(miss)}")
            return
        df.dropna(subset=["food_name"], inplace=True)
        df["food_name"] = df["food_name"].astype(str)
        df["type"] = df["type"].astype(str).fillna("")
        if args.max_test_img_num != 0:
            df = df.iloc[: args.max_test_img_num]
    except Exception as e:
        print(f"[ERR] load excel: {e}")
        return

    logger = IntervalLogger(
        "GPU Memory",
        target_gpus=[0, 1, 2, 3, 4, 5, 6, 7],
        interval=2.0,
        level="DEBUG",
        color="RED",
    )
    logger.start()

    # 加载模型与处理器（Torch / HF）
    print(f"Loading model from {args.model_path} ...")
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None

    device_map, max_memory = select_devices_and_max_memory(
        tp_size=args.tensor_parallel_size, util=args.gpu_memory_utilization
    )

    load_kwargs = dict(
        dtype=dtype,
        trust_remote_code=True,
    )
    # 只有在有 GPU 时才传 device_map/max_memory
    if device_map is not None:
        load_kwargs.update(device_map=device_map, max_memory=max_memory)
    
    try:
        model_map = {
            'Qwen3-VL-4B-Instruct': Qwen3VLForConditionalGeneration,
            'Qwen3-VL-8B-Instruct': Qwen3VLForConditionalGeneration,
            'Qwen3-VL-30B-A3B-Instruct': Qwen3VLMoeForConditionalGeneration,
            'Qwen3-VL-235B-A22B-Instruct': Qwen3VLMoeForConditionalGeneration,
        }
        model_name = args.model_path.rstrip('/').split('/')[-1]
        model = model_map.get(model_name, AutoModel).from_pretrained(
            args.model_path, **load_kwargs
        )
    except Exception as e1:
        print(
            f"[WARN] Qwen3VLMoeForConditionalGeneration failed, fallback to AutoModel: {e1}"
        )
        model = AutoModel.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # 让 tokenizer 的最大长度对齐 vLLM 的 max_model_len（文本侧截断）
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = args.max_model_len

    # 可选：指定注意力实现
    if args.attn_impl is not None and hasattr(model, "config"):
        try:
            model.config.attn_implementation = args.attn_impl
        except Exception:
            pass

    # 评估
    evaluate_model_accuracy(
        model_path=args.model_path,
        samples_df=df,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_path,
        img_root=args.input_img_path,
        image_max_long=args.image_max_long,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        args=args,
    )

    logger.stop()
    logger.join(timeout=1.0)


if __name__ == "__main__":
    main()
