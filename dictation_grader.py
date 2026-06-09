"""
英语单词听写批量批改工具
直接上传学生听写纸图片，AI 自动判断每道题翻译是否正确

用法: python dictation_grader.py <听写纸图片>
      python dictation_grader.py <听写纸图片> --cols 3 --per-col 33

示例: python dictation_grader.py paper1.jpg
"""

import base64
import io
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("请先安装 Pillow: pip install Pillow")
    sys.exit(1)

# 自动加载 .env 文件
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = val.strip()

PROVIDERS = {
    "qwen": {
        "name": "通义千问 Qwen-VL",
        "api_key_env": "DASHSCOPE_API_KEY",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-plus",
    },
    "glm": {
        "name": "智谱 GLM-4V",
        "api_key_env": "GLM_API_KEY",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
    },
}


class OpenAIClient:
    def __init__(self, provider):
        from openai import OpenAI
        self.name = provider["name"]
        self.model = provider["model"]
        api_key = os.environ.get(provider["api_key_env"])
        if not api_key:
            raise ValueError(f"请设置环境变量 {provider['api_key_env']}")
        self.client = OpenAI(api_key=api_key, base_url=provider["api_base"])

    def chat(self, image_b64, prompt, max_tokens=8192):
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content


def create_client(provider_name):
    info = PROVIDERS.get(provider_name)
    if not info:
        print(f"不支持的 provider: {provider_name}，可选: {list(PROVIDERS.keys())}")
        sys.exit(1)
    return OpenAIClient(info)


def encode_image_pil(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def split_columns(image_path, num_cols):
    """将图片水平均分为 N 列"""
    img = Image.open(image_path)
    w, h = img.size
    col_w = w // num_cols
    cols = []
    for i in range(num_cols):
        left = i * col_w
        right = (i + 1) * col_w if i < num_cols - 1 else w
        col = img.crop((left, 0, right, h))
        cols.append(col)
    return cols, (w, h)


def build_prompt(col_index, num_cols, per_col):
    start = col_index * per_col + 1
    end = start + per_col - 1
    return f"""你是一位英语老师，正在批改学生的英语单词听写。

这是一张完整听写纸的第 {col_index + 1}/{num_cols} 列，包含第 {start} 题到第 {end} 题。
每道题左侧印刷了英文单词，右侧学生手写了中文翻译。

请逐一查看图片中每一道题：
1. 看清楚印刷的英文单词
2. 看清楚学生手写的中文翻译
3. 根据你的知识判断翻译是否正确

判断规则：
- 翻译意思正确 → "对"（近义词、同义词都算对）
- 翻译明显错误 → "错"
- 看不清、空白或无法辨认 → "存疑"

重要：
- 如果图片根本不是听写纸、完全看不清内容、或者没有任何题目信息，返回空数组 []
- 不要编造任何内容。只输出你确实能看到的信息
- 实在看不清的内容，判定必须为"存疑"，学生写了填"看不清"
- 每道题必须有明确的英文印刷字和手写中文痕迹，两者都缺就整张判为无效

返回纯 JSON 数组（不要 markdown 代码块，不要其他文字），格式：
[
  {{"题号": {start}, "英文": "take action", "正确翻译": "采取行动", "学生写了": "采取行动", "判定": "对"}},
  ...
]

"正确翻译"填你认为的正确翻译，"学生写了"填你实际看到的内容。
题号从 {start} 开始，到 {end} 结束。只输出你能确认存在的题。"""


def build_full_page_prompt():
    return """你是一位英语老师，正在批改学生的英语单词听写。

听写纸格式：A4 纸横向 3 大列，每列 33 题，共 99 题。每道题印刷了英文单词，学生手写了中文翻译。

请逐一查看图片中全部 99 道题：
1. 看清楚印刷的英文单词
2. 看清楚学生手写的中文翻译
3. 根据你的知识判断翻译是否正确

判断规则：
- 翻译意思正确 -> "对"（近义词、同义词都算对）
- 翻译明显错误 -> "错"
- 看不清、空白或无法辨认 -> "存疑"

返回纯 JSON 数组（不要 markdown 代码块，不要其他文字），包含全部 99 道题，格式：
[
  {"题号": 1, "英文": "take action", "正确翻译": "采取行动", "学生写了": "采取行动", "判定": "对"},
  ...
  {"题号": 99, "英文": "...", "正确翻译": "...", "学生写了": "...", "判定": "..."}
]

"正确翻译"填你认为的正确翻译，"学生写了"填你实际看到的内容。
题号从 1 到 99，一道不许漏。仔细看完 3 列全部内容再输出。"""


def print_summary(results):
    results = sorted(results, key=lambda r: r["题号"])
    total = len(results)
    correct = sum(1 for r in results if r["判定"] == "对")
    wrong = sum(1 for r in results if r["判定"] == "错")
    uncertain = sum(1 for r in results if r["判定"] == "存疑")

    print(f"\n{'='*55}")
    print(f"  批改完成: 共 {total} 题 | 对 {correct} | 错 {wrong} | 存疑 {uncertain}")
    if total > 0:
        print(f"  正确率: {correct}/{total} = {correct/total*100:.1f}%  |  需复核: {uncertain} 题")
    print(f"{'='*55}")

    if wrong > 0:
        print("\n【错题】")
        for r in results:
            if r["判定"] == "错":
                print(f"  第{r['题号']:>2}题 {r['英文']:<25} 正确「{r['正确翻译']}」 学生写了「{r['学生写了']}」")

    if uncertain > 0:
        print(f"\n【存疑，需人工复核】")
        for r in results:
            if r["判定"] == "存疑":
                print(f"  第{r['题号']:>2}题 {r['英文']:<25} 正确「{r['正确翻译']}」")


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def main():
    import argparse
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="英语单词听写批量批改工具")
    parser.add_argument("image", help="听写纸图片路径")
    parser.add_argument("--no-split", action="store_true", help="不切图，整张送入")
    parser.add_argument("--provider", "-p", default="qwen", choices=list(PROVIDERS.keys()),
                        help="AI 服务商（默认 qwen）")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"错误: 图片不存在 - {args.image}")
        sys.exit(1)

    client = create_client(args.provider)
    start_time = time.time()

    if args.no_split:
        # 整张图直接送
        print("整张图模式: 1 次调用")
        img = Image.open(args.image)
        image_b64 = encode_image_pil(img)
        prompt = build_full_page_prompt()
        raw = client.chat(image_b64, prompt)
        results = extract_json(raw)
    else:
        # 切 3 列，并行发送
        columns, (w, h) = split_columns(args.image, 3)
        print(f"并行切图模式: 3 列同时发送, 每列约 33 题 (原图 {w}x{h})")

        def grade_column(i, col_img):
            start = i * 33 + 1
            end = (i + 1) * 33
            image_b64 = encode_image_pil(col_img)
            prompt = build_prompt(i, 3, 33)
            for attempt in range(2):
                try:
                    raw = client.chat(image_b64, prompt)
                    return extract_json(raw)
                except Exception as e:
                    if attempt == 0:
                        print(f"  [列 {i+1}] 出错，重试... ({e})")
                    else:
                        print(f"  [列 {i+1}] 重试仍失败: {e}")
                        return []

        all_results = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(grade_column, i, col): i for i, col in enumerate(columns)}
            for future in as_completed(futures):
                i = futures[future]
                results = future.result()
                print(f"  第 {i+1}/3 列完成: {len(results)} 题")
                all_results.extend(results)

        results = all_results

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.0f} 秒")

    print_summary(results)

    out_path = Path(args.image).stem + "_批改结果.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
