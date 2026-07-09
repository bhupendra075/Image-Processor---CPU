# python -m venv .venv
# .\.venv\Scripts\Activate.ps1
# pip install -r requirements.txt
 # uvicorn app:app --host 127.0.0.1 --port 8000 --reload


import os
import time
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
import requests

BASE = "http://127.0.0.1:8000/process"
SIZES = [512, 1024, 2048]
FILTERS = ["blur", "sobel"]
REPS = 3
CONCURRENCY = 4
OUTDIR = "test_results"

os.makedirs(OUTDIR, exist_ok=True)

results = {"requests": [], "direct": []}


def make_test_image(size):
    # Create a grayscale test image with gradients + noise
    x = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.tile(x, (size, 1))
    noise = (np.random.randn(size, size) * 10).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(noisy).save(buf, format="PNG")
    return buf.getvalue(), noisy


def post_image(img_bytes, filter_type, kernel_size=3):
    files = {"file": ("img.png", img_bytes, "image/png")}
    data = {"filter_type": filter_type}
    if filter_type == "blur":
        data["kernel_size"] = str(kernel_size)
    start = time.perf_counter()
    r = requests.post(BASE, files=files, data=data, timeout=60)
    total = time.perf_counter() - start
    r.raise_for_status()
    header = r.headers.get("X-Processing-Metrics") or r.headers.get("x-processing-metrics")
    return total, header, r.content


# Run networked concurrent tests
for size in SIZES:
    img_bytes, arr = make_test_image(size)
    for filt in FILTERS:
        tasks = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = []
            for i in range(REPS):
                # For blur, vary kernel size modestly
                ks = 5 if filt == "blur" else 3
                futures.append(ex.submit(post_image, img_bytes, filt, ks))
            for idx, fut in enumerate(as_completed(futures)):
                try:
                    total, header, content = fut.result()
                    fname = os.path.join(OUTDIR, f"out_{filt}_{size}_{idx}.png")
                    with open(fname, "wb") as f:
                        f.write(content)
                    res = {"mode": "network", "filter": filt, "size": size, "total_seconds": total, "header": header, "out_file": fname}
                    print(json.dumps(res))
                    results["requests"].append(res)
                except Exception as e:
                    print("Request failed:", e)

# Direct (in-process) function timing of fallback implementations for baseline
try:
    import app
    print("Running direct in-process fallback timings...")
    for size in SIZES:
        _, arr = make_test_image(size)
        a = arr.copy()
        start = time.perf_counter()
        # choose blur fallback if compiled module not available
        if hasattr(app, 'box_blur_py'):
            out = app.box_blur_py(a, 5)
            dur = time.perf_counter() - start
            fname = os.path.join(OUTDIR, f"direct_blur_{size}.png")
            Image.fromarray(out).save(fname)
            results["direct"].append({"mode": "direct", "filter": "blur", "size": size, "seconds": dur, "out_file": fname})
        # sobel
        a = arr.copy()
        start = time.perf_counter()
        if hasattr(app, 'sobel_edge_py'):
            out2 = app.sobel_edge_py(a)
            dur2 = time.perf_counter() - start
            fname2 = os.path.join(OUTDIR, f"direct_sobel_{size}.png")
            Image.fromarray(out2).save(fname2)
            results["direct"].append({"mode": "direct", "filter": "sobel", "size": size, "seconds": dur2, "out_file": fname2})
except Exception as e:
    print("Skipping direct in-process tests (app import failed):", e)

# Save metrics
with open(os.path.join(OUTDIR, "metrics.json"), "w") as mf:
    json.dump(results, mf, indent=2)

print("Done. Results saved to:", OUTDIR)
