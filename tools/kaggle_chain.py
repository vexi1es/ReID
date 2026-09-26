# ЛЦТ#7: цепочка экспериментов на Kaggle без участия ноутбука.
# Две очереди параллельно — по одной на каждую T4 (GPU T4 x2). Каждый эксперимент -> свой zip в Output.
import glob
import json
import os
import shutil
import subprocess
import threading
import time
import zipfile

CHAIN = os.environ.get("CHAIN", "__CHAIN__")
VIT = dict(backbone="facebook/dinov2-base", size=224, freeze=6, lr=3e-5, P=12, K=4)
CNX = dict(backbone="timm:convnext_small.dinov3_lvd1689m", size=256, freeze=2, lr=2e-5, P=12, K=4)
PLAN = {
    "A": [  # GPU0, GPU1
        [dict(VIT, name="vitB_P16", P=16, seed=0), dict(VIT, name="vitB_seed1_full", seed=1, full=True), dict(VIT, name="vitB_seed3", seed=3)],
        [dict(VIT, name="vitB_seed2", seed=2), dict(VIT, name="vitB_seed2_full", seed=2, full=True), dict(VIT, name="vitB_seed3_full", seed=3, full=True)],
    ],
    "B": [
        [dict(CNX, name="cnxS_seed2", seed=2), dict(CNX, name="cnxS_seed1_full", seed=1, full=True), dict(CNX, name="cnxS_seed3", seed=3)],
        [dict(CNX, name="cnxS_320_full", size=320, seed=0, full=True), dict(CNX, name="cnxS_seed2_full", seed=2, full=True), dict(CNX, name="cnxS_seed3_full", seed=3, full=True)],
    ],
}[CHAIN]

subprocess.run("pip install -q -U 'timm>=1.0.20'", shell=True)
src = os.path.dirname(glob.glob("/kaggle/input/**/code/train.py", recursive=True)[0])
root = os.path.dirname(src)
work = "/kaggle/working/lct7"
if not os.path.exists(work):
    shutil.copytree(src, work)
os.chdir(work)
t = open("train.py").read().replace("torch.cuda.is_bf16_supported()", "torch.cuda.get_device_capability()[0] >= 8")
open("train.py", "w").write(t)
DATA, CROPS = f"{root}/data", f"{root}/crops"
LOG = "/kaggle/working/progress.txt"


def note(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def run_exp(E, gpu):
    out = f"runs/{E['name']}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = (f"python -u train.py --data '{DATA}' --crops '{CROPS}' --out {out} --workers 3 "
           f"--backbone {E['backbone']} --size {E['size']} --freeze {E['freeze']} --lr {E['lr']} "
           f"--P {E['P']} --K {E['K']} --seed {E['seed']} --epochs 20 --eval-every 4"
           + (" --full" if E.get("full") else ""))
    note(f"GPU{gpu} старт {E['name']}")
    with open(f"{out}/train.log", "w") as f:
        rc = subprocess.run(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(f"{out}/log.json"):
        note(f"GPU{gpu} СБОЙ {E['name']} rc={rc}")
        return
    if not E.get("full"):
        subprocess.run(f"python eval_run.py --data '{DATA}' --crops '{CROPS}' --runs {out} --out {out}/eval.json "
                       f"> {out}/eval.log 2>&1", shell=True, env=env)
    z = f"/kaggle/working/result_{E['name']}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:
        ck = "best.pt" if not E.get("full") else "last.pt"
        for n in (ck, "log.json", "val_emb.npy", "eval.json", "train.log"):
            if os.path.exists(f"{out}/{n}"):
                f.write(f"{out}/{n}", f"{E['name']}/{n}")
    for n in ("best.pt", "last.pt", "state.pt"):          # место в /kaggle/working ограничено
        if os.path.exists(f"{out}/{n}"):
            os.remove(f"{out}/{n}")
    res = ""
    if os.path.exists(f"{out}/eval.json"):
        res = f" итог mAP {json.load(open(f'{out}/eval.json'))['final']['mAP']:.4f}"
    note(f"GPU{gpu} готово {E['name']}{res}")


def lane(gpu, exps):
    for E in exps:
        try:
            run_exp(E, gpu)
        except Exception as e:
            note(f"GPU{gpu} ошибка {E['name']}: {e}")


import torch  # noqa: E402

n = torch.cuda.device_count()
note(f"цепочка {CHAIN}, GPU: {n}")
if n >= 2:
    th = [threading.Thread(target=lane, args=(i, PLAN[i])) for i in range(2)]
    [x.start() for x in th]
    [x.join() for x in th]
else:
    lane(0, PLAN[0] + PLAN[1])
shutil.rmtree("/kaggle/working/lct7", ignore_errors=True)   # в Output — только zip-ы и progress.txt
note("ВСЁ")
