import re


# 仅移除生成文本中的图片占位符，不影响 <think> 内容
_IMG_TAG = re.compile(r"(?:<\|image\|>)+", re.S)
# 捕获思考段（用于单独保存思考内容）
_THINK_CAP = re.compile(r"<think>(.*?)</think>", re.S)
# 捕获答案部分
# _BOX_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.S)
_BOX_RE = re.compile(r"\\boxed{(?:\\text{)?(.*?)(?:})?}?", re.S)


def exact_match(pred: str, gt: str) -> bool:
    """
    最简单精确匹配，你可以按需替换为更鲁棒的比较（去标点/空格/大小写等）。
    """
    return pred.strip() == gt.strip()


def extract_final_cn(s: str, *, strip_edge_punct: bool = False) -> str:

    # 中文汉字 + 常见中文标点（含全角），另补 U+00B7 ·、U+2014 —、U+2026 …
    _CN_BLOCKS = (
        r"\u3400-\u4DBF"  # CJK Ext A
        r"\u4E00-\u9FFF"  # CJK Unified Ideographs
        r"\uF900-\uFAFF"  # CJK Compatibility Ideographs
        r"\u3000-\u303F"  # CJK Symbols & Punctuation（。、“”、——、…… 等）
        r"\uFF01-\uFF0F"  # 全角标点片段
        r"\uFF1A-\uFF20"
        r"\uFF3B-\uFF40"
        r"\uFF5B-\uFF65"
        r"\u00B7"  # ·
        r"\u2014"  # —
        r"\u2026"  # …
    )

    # 1) 抓“最后一对花括号”里的中文：   {...中文...}，要求此处的 '}' 之后**不再出现 '{'**
    #    这样能拿到 \text{中文} 这种**内层**的中文（通常就是我们要的）。
    _BRACED_LAST_CN = re.compile(r"\{([" + _CN_BLOCKS + r"]+)\}(?!.*\{)", re.S)

    # 2) 兜底：整串里“最后一段中文”
    _LAST_CN_SPAN = re.compile(r"[" + _CN_BLOCKS + r"]+")

    s = (s or "").strip()
    m = _BRACED_LAST_CN.search(s)
    if m:
        out = m.group(1)
    else:
        ms = list(_LAST_CN_SPAN.finditer(s))
        out = ms[-1].group(0) if ms else ""

    if strip_edge_punct and out:
        # 如需去掉首尾中文标点（保留中间的），打开这个选项
        edge_punct = re.compile(
            r"^[" + _CN_BLOCKS + r"&&\p{P}]*|[" + _CN_BLOCKS + r"&&\p{P}]*$"
        )
        try:
            import regex  # 可选：如果安装了 regex，可以更精准判断标点

            out = regex.sub(r"^\p{P}+", "", out)
            out = regex.sub(r"\p{P}+$", "", out)
        except Exception:
            # 退化：常见尾句号/逗号等
            out = re.sub(r"^[，。、“”‘’！？：；《》【】…—·]+", "", out)
            out = re.sub(r"[，。、“”‘’！？：；《》【】…—·]+$", "", out)
    return out