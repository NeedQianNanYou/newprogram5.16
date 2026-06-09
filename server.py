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

# Config
env_path = Path(__file__).parent / ".env"
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

FRONTEND = (Path(__file__).parent / "static" / "review.html").read_text(encoding="utf-8")

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
    prompt = f"""你是英语批改员。这是听写纸第{i+1}/3列，题号{s}到{e}，印刷英文单词，右侧学生手写中文翻译。

【关键要求】
1. 逐字辨认手写中文——横竖撇捺都要看清，不要一眼扫过就抄正确翻译
2. 学生写的≠正确翻译。先看学生写了什么，再判断对错
3. 潦草字要仔细辨认，宁可标"存疑"不要猜
4. 禁止因为认识英文单词就直接脑补正确翻译到"学生写了"

【判题标准】
学生写的中文和英文单词真实意思一致→对对"
明显错误→"错"
看不清/空白→"存疑"

返回纯JSON：[{{"题号":{s},"英文":"...","正确翻译":"...","学生写了":"...","判定":"对/错/存疑"}}]"""
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
            return data
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
