#!/usr/bin/env python3
"""Generate qwen_golden.json: reference prompts for the Qwen chat-template models.

Each case is a Chat Completions request. It is converted to template messages
the way vLLM does it (tool-call arguments parsed into objects, null content as
an empty string, images and videos as {"type": ...} parts), rendered with the
model's chat template by Hugging Face transformers' apply_chat_template, and
tokenized by the reference tokenizer. Images and videos are then expanded
token by token, as vLLM's Qwen3-VL processor does: an image's <|image_pad|>
becomes its patch tokens, a video's <|vision_start|><|video_pad|><|vision_end|>
becomes one "<t seconds>" + <|vision_start|> + patches + <|vision_end|> group
per temporal patch. Every patch position holds the item's pad value
(2^30 + (BE_u64(sha256(reference)[:8]) & (2^30 - 1))).

The planning functions below are copies of transformers' Qwen2-VL image and
Qwen3-VL video processing (smart_resize, sample_frames, timestamps) with the
per-frame pixel cap of the reference implementation (qwen-vl-utils).

    python3 tokenizer/testdata/gen_qwen_golden.py   # from the repository root
"""

import copy
import hashlib
import json
import math
import os

import numpy as np
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "assets", "tokenizers")


def load(name):
    d = os.path.join(ASSETS, name)
    tok = AutoTokenizer.from_pretrained(os.path.join(ASSETS, "qwen3"))
    cfg = json.load(open(os.path.join(d, "tokenizer_config.json")))
    tok.chat_template = cfg["chat_template"]
    return tok


TOK = {"qwen3": load("qwen3"), "qwen3-vl": load("qwen3-vl")}

# ---------------------------------------------------------------- planning

PATCH, MERGE, TPATCH = 16, 2, 2
FACTOR = PATCH * MERGE
IMAGE_MIN, IMAGE_MAX = 65536, 16777216
VIDEO_MIN, VIDEO_MAX = 4096, 25165824
VIDEO_FPS, MIN_FRAMES, MAX_FRAMES, MAX_FRAME_TOKENS = 2.0, 4, 768, 768


def image_smart_resize(height, width, factor=FACTOR, min_pixels=IMAGE_MIN, max_pixels=IMAGE_MAX):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def video_smart_resize(num_frames, height, width, temporal_factor=TPATCH, factor=FACTOR, min_pixels=VIDEO_MIN, max_pixels=VIDEO_MAX):
    if num_frames < temporal_factor:
        raise ValueError("frames")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def sample_frames(total, video_fps, fps=None, num_frames=None):
    if num_frames is None:
        fps = VIDEO_FPS if fps is None else fps
        num_frames = int(total / video_fps * fps)
        num_frames = min(max(num_frames, MIN_FRAMES), MAX_FRAMES, total)
    return np.linspace(0, total - 1, num_frames).round().astype(int).tolist()


def timestamps(indices, video_fps, merge=TPATCH):
    indices = list(indices)
    if len(indices) % merge != 0:
        indices.extend(indices[-1] for _ in range(merge - len(indices) % merge))
    ts = [idx / video_fps for idx in indices]
    return [(ts[i] + ts[i + merge - 1]) / 2 for i in range(0, len(ts), merge)]


def plan_image(meta):
    h, w = image_smart_resize(meta["height"], meta["width"])
    return {"width": w, "height": h, "tokens": (h // PATCH) * (w // PATCH) // (MERGE * MERGE)}


def plan_video(meta, opts):
    total = meta.get("total_frames") or round(meta["duration_ms"] / 1000 * meta["video_fps"])
    idx = sample_frames(total, meta["video_fps"], fps=opts.get("fps"), num_frames=opts.get("nframes"))
    n = len(idx)
    frame_cap = MAX_FRAME_TOKENS * FACTOR * FACTOR
    per_frame = max(min(frame_cap, VIDEO_MAX // n), int(VIDEO_MIN * 1.05))
    h, w = video_smart_resize(n, meta["height"], meta["width"], max_pixels=per_frame * n)
    return {"width": w, "height": h, "tokens": (h // PATCH) * (w // PATCH) // (MERGE * MERGE),
            "nframes": n, "kept_frames": (n + (-n % TPATCH)) // TPATCH, "timestamps": timestamps(idx, meta["video_fps"])}


def pad_value(reference):
    h = hashlib.sha256(reference.encode()).digest()
    return (1 << 30) + (int.from_bytes(h[:8], "big") & ((1 << 30) - 1))


# ------------------------------------------------------- request conversion


def to_template_messages(body, media):
    out = []
    for m in body["messages"]:
        m = copy.deepcopy(m)
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p["type"] == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p["type"] in ("image_url", "video_url"):
                    kind = "image" if p["type"] == "image_url" else "video"
                    media.append((kind, p))
                    parts.append({"type": kind})
            m["content"] = parts
        elif content is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args)
        out.append(m)
    return out


def render(model, body):
    tok = TOK[model]
    media = []
    messages = to_template_messages(body, media)
    kwargs = dict(body.get("chat_template_kwargs") or {})
    cont = body.get("continue_final_message", False)
    text = tok.apply_chat_template(
        messages, tools=body.get("tools"), tokenize=False, add_generation_prompt=not cont,
        continue_final_message=cont, **kwargs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    expanded = []
    plans = []
    mi = 0
    i = 0
    vs, vp, ve, ip = (tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|video_pad|>", "<|vision_end|>", "<|image_pad|>"))
    while i < len(ids):
        t = ids[i]
        if t == ip:
            kind, part = media[mi]
            assert kind == "image"
            mi += 1
            plan = plan_image(part["metadata"])
            plans.append(plan)
            expanded += [pad_value(part["metadata"]["digest"])] * plan["tokens"]
            i += 1
            continue
        if t == vs and i + 2 < len(ids) and ids[i + 1] == vp and ids[i + 2] == ve:
            kind, part = media[mi]
            assert kind == "video"
            mi += 1
            plan = plan_video(part["metadata"], part.get("frame_extract_options") or {})
            plans.append(plan)
            for k, ts in enumerate(plan["timestamps"]):
                expanded += tok(f"<{ts:.1f} seconds>", add_special_tokens=False)["input_ids"]
                expanded += [vs] + [pad_value(f"{part['metadata']['digest']}_kf_{k:04}")] * plan["tokens"] + [ve]
            i += 3
            continue
        expanded.append(t)
        i += 1
    assert mi == len(media)
    case = {"text": text, "token_ids": ids}
    if media:
        case["expanded_token_ids"] = expanded
        case["plans"] = plans
    return case


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather by city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市"}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search_docs", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}

CASES = [
    ("qwen3", "system_user_thinking", {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello"}],
                                       "chat_template_kwargs": {"enable_thinking": True}}),
    ("qwen3", "system_user_no_thinking", {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello"}],
                                          "chat_template_kwargs": {"enable_thinking": False}}),
    ("qwen3", "tools_no_thinking", {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER, SEARCH],
                                    "chat_template_kwargs": {"enable_thinking": False}}),
    ("qwen3", "agent_loop_thinking", {"messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "What's the weather in 广州 and Paris?"},
        {"role": "assistant", "content": None, "reasoning_content": "I should call the weather tool twice.",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"广州\"}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 30°C"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Rain, 12°C"}],
        "tools": [WEATHER], "chat_template_kwargs": {"enable_thinking": True}}),
    ("qwen3", "history_drops_old_reasoning", {"messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple arithmetic."},
        {"role": "user", "content": "And 3+3? Answer   with  spaces\n\nand newlines 🙂"}],
        "chat_template_kwargs": {"enable_thinking": True}}),
    ("qwen3", "continue_final_message", {"messages": [
        {"role": "user", "content": "Count to three."}, {"role": "assistant", "content": "One, two,"}],
        "continue_final_message": True, "chat_template_kwargs": {"enable_thinking": False}}),
    ("qwen3-vl", "image_and_text", {"messages": [
        {"role": "system", "content": "You describe images."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://img.example/a.jpg"},
             "metadata": {"digest": digest("a"), "width": 1920, "height": 1080}},
            {"type": "text", "text": "What is in the image?"}]}]}),
    ("qwen3-vl", "video_image_tools", {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Compare the clip "},
            {"type": "video_url", "video_url": {"url": "https://video.example/c.mp4"},
             "metadata": {"digest": digest("clip"), "width": 1280, "height": 720, "duration_ms": 10000, "total_frames": 300, "video_fps": 30},
             "frame_extract_options": {"fps": 1}},
            {"type": "text", "text": " with this picture: "},
            {"type": "image_url", "image_url": {"url": "https://img.example/b.png"},
             "metadata": {"digest": digest("b"), "width": 300, "height": 200}}]}],
        "tools": [SEARCH]}),
    ("qwen3-vl", "small_and_odd_videos", {"messages": [
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "https://video.example/s.mp4"},
             "metadata": {"digest": digest("short"), "width": 640, "height": 360, "duration_ms": 2500, "video_fps": 24},
             "frame_extract_options": {"nframes": 5}},
            {"type": "image_url", "image_url": {"url": "https://img.example/t.png"},
             "metadata": {"digest": digest("tiny"), "width": 20, "height": 30}},
            {"type": "text", "text": "Describe."}]}]}),
]


def main():
    cases = []
    for model, name, body in CASES:
        out = render(model, body)
        cases.append({"name": name, "model": model, "request": body, **out})
        print(f"{name}: {len(out['token_ids'])} tokens" + (f", {len(out['expanded_token_ids'])} expanded" if "expanded_token_ids" in out else ""))
    # Timestamp texts as the tokenizer formats them (Python {:.1f}).
    stamps = {f"{t:.1f}": TOK["qwen3"](f"<{t:.1f} seconds>", add_special_tokens=False)["input_ids"]
              for t in [0.0, 0.25, 0.35, 1.25, 2.45, 9.95, 123.45]}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_golden.json")
    with open(path, "w") as f:
        json.dump({"generated_by": "tokenizer/testdata/gen_qwen_golden.py",
                   "transformers": __import__("transformers").__version__,
                   "cases": cases, "timestamp_texts": stamps}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
