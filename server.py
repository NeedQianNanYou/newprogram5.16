"""
听写批改 - AI初筛+教师复核
Qwen视觉判题 → 对的自动过 → 错/存疑交老师复核
"""
import base64, io, json, os, uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import uvicorn

# Config: 兼容 PyInstaller 打包和源码运行
import sys
if getattr(sys, 'frozen', False):
    BASE = Path(sys._MEIPASS)  # PyInstaller
else:
    BASE = Path(__file__).parent  # 源码运行

env_path = BASE / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
MODEL = "qwen-vl-plus"
client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

FRONTEND = (BASE / "static" / "review.html").read_text(encoding="utf-8")

app = FastAPI()


def b64(img):
    # 预处理：增强手写字迹可读性
    img = img.convert("L")  # 灰度化
    img = ImageEnhance.Contrast(img).enhance(2.0)  # 增强对比度
    img = ImageEnhance.Sharpness(img).enhance(2.0)  # 锐化
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150))  # 去模糊
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def split_cols(img, n=3):
    w, h = img.size; cw = w // n
    return [img.crop((i*cw, 0, (i+1)*cw if i<n-1 else w, h)) for i in range(n)]


def grade_col(i, col_img):
    b = b64(col_img)
    s, e = i*33+1, (i+1)*33
    prompt = f"""你是英语听写批改员。严格按照以下两步执行：

【步骤1：逐字识别】先不看答案，逐题仔细看图片，依次读出：
- 题号（印刷数字）
- 英文单词（印刷体，如实抄下来）
- 学生手写的中文（一个字一个字辨认，写的是什么就记录什么。如果看不清，填"看不清"）

【步骤2：翻译对比】对于步骤1中读出的每道题，用你的知识给出英文单词的正确中文翻译，然后和步骤1中读出的学生手写内容对比：
- 意思一致或相近→"对"
- 明显不同→"错"
- 学生留空或看不清→"存疑"

【核心禁止事项】
- 严禁把"正确翻译"直接抄到"学生写了"！！学生写的是图片中手写的内容，正确翻译是你大脑中的知识，这是两回事
- 如果图片不是听写纸，返回[]

返回纯JSON：[{{"题号":{s},"英文":"...","正确翻译":"...","学生写了":"...","判定":"..."}}]，题号范围{s}到{e}"""
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=8192,
                messages=[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}},
                    {"type":"text","text":prompt}]}])
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
            data = json.loads(raw)
            data = [d for d in data if s <= d.get("题号",0) <= e]
            seen = set(); uniq = []
            for d in data:
                if d["题号"] not in seen: seen.add(d["题号"]); uniq.append(d)
            return uniq
        except Exception as ex:
            if attempt == 0: continue
            print(f"列{i+1}失败:{ex}"); return []


def grade_paper(path):
    img = Image.open(path)
    cols = split_cols(img, 3)
    all_r = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs = {pool.submit(grade_col, i, c): i for i, c in enumerate(cols)}
        for f in as_completed(fs):
            try: all_r.extend(f.result())
            except Exception as e: print(f"列{fs[f]+1}:{e}")

    # 后处理：学生明显没写的（空白/看不清），自动判错
    for r in all_r:
        s = (r.get("学生写了") or "").strip()
        if not s or s in ("看不清", "(空白)", ""):
            r["判定"] = "错"
            r["学生写了"] = "识别不清"
    return all_r


@app.get("/")
async def index():
    return HTMLResponse(FRONTEND)


@app.post("/api/grade")
async def grade(files: list[UploadFile] = File(...)):
    ud = Path(__file__).parent / "uploads"; ud.mkdir(exist_ok=True)
    saved = []
    for f in files:
        p = ud / f"{uuid.uuid4().hex[:8]}.jpg"
        p.write_bytes(await f.read())
        saved.append((f.filename or "unknown", str(p)))

    all_r = {}; errs = []
    def do(item):
        nm, p = item
        try: return Path(nm).stem, grade_paper(p), None
        except Exception as e: return Path(nm).stem, [], str(e)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fs = [pool.submit(do, item) for item in saved]
        for f in as_completed(fs):
            nm, res, e = f.result()
            if e: errs.append({"name":nm,"error":e})
            else: all_r[nm] = res

    summary = []
    for nm, res in sorted(all_r.items()):
        t = len(res)
        c = sum(1 for r in res if r["判定"]=="对")
        w = sum(1 for r in res if r["判定"]=="错")
        u = sum(1 for r in res if r["判定"]=="存疑")
        summary.append({"name":nm,"total":t,"correct":c,"wrong":w,"uncertain":u,
                       "rate":f"{c/t*100:.1f}%" if t else "0%","details":res})

    return JSONResponse({"success":True,"summary":summary,"errors":errs})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
