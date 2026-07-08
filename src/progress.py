"""Pipeline ilerleme ozeti: egitim (train_ce.log) + test inference (infer_ce.log).

Eski surumun kusuru: yalnizca egitim adim satirlarini okuyordu. Fold 2'nin son
adimindan sonra OOF/proxy skorlama ve ayri loga yazan 3 saatlik inference fazi
boyunca ekran %99'da donmus gibi gorunuyordu. Bu surum tum fazlari izler ve
logun bayatlayip bayatlamadigini (gercek takilma sinyali) ayrica raporlar.

Kullanim: python progress.py          (tek seferlik ozet)
          python progress.py --watch  (30 sn'de bir yeniler)
"""
import re
import sys
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import config as C

_V = next((a for a in sys.argv[1:] if re.fullmatch(r"v\d+", a)), "")
SUFFIX = f"_{_V}" if _V else ""
TRAIN_LOG = C.ARTIFACTS_DIR / f"train_ce{SUFFIX}.log"
INFER_LOG = C.ARTIFACTS_DIR / f"infer_ce{SUFFIX}.log"
# v4 = tek full-data model (fold yok); digerleri 3-fold
N_FOLDS = 1 if _V == "v4" else C.CE_N_FOLDS

STEP_RE = re.compile(r"fold (\d+) e(\d+) step (\d+)/(\d+) loss=([\d.]+) \((\d+)s\)")
OOF_PROG_RE = re.compile(r"oof f(\d+) (\d+)/(\d+)")
OOF_DONE_RE = re.compile(r"fold (\d+) OOF macro_f1@0\.5 = ([\d.]+)")
PROXY_PROG_RE = re.compile(r"proxy f(\d+) (\d+)/(\d+)")
TRAIN_FINAL_RE = re.compile(r"\[CE(?: v\d+)? LB-proxy\].*en_iyi_thr=([\d.]+) f1=([\d.]+)")
TEST_PROG_RE = re.compile(r"test f(\d+) (\d+)/(\d+)")
TEST_DONE_RE = re.compile(r"fold (\d+) test skorlandi")
INFER_FINAL_RE = re.compile(r"yazildi: (\S+)")

# fold ici zaman paylari (gozlenen kosu: egitim ~7400s, OOF ~300s, proxy ~200s)
W_TRAIN, W_OOF, W_PROXY = 0.93, 0.04, 0.03

ACTIVITY_RES = (STEP_RE, OOF_PROG_RE, PROXY_PROG_RE, OOF_DONE_RE,
                TEST_PROG_RE, TEST_DONE_RE)


def _read(path):
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _last(pattern, text, fold=None):
    last = None
    for m in pattern.finditer(text):
        if fold is None or int(m.group(1)) == fold:
            last = m
    return last


def _last_activity(text):
    for line in reversed(text.splitlines()):
        for r in ACTIVITY_RES:
            if r.search(line):
                return line.strip()
    return ""


def _fold_trained(text, f):
    return re.search(rf"fold {f}(?: egitildi|: checkpoint mevcut)", text) is not None


def _train_frac(text):
    """Egitim scriptinin genel ilerlemesi (egitim + OOF + proxy fazlari dahil)."""
    n = N_FOLDS
    total = 0.0
    for f in range(n):
        frac = 0.0
        if _fold_trained(text, f):
            frac = W_TRAIN
        else:
            m = _last(STEP_RE, text, f)
            if m:
                frac = W_TRAIN * int(m.group(3)) / int(m.group(4))
        if re.search(rf"fold {f} OOF macro_f1", text):
            frac += W_OOF
        else:
            m = _last(OOF_PROG_RE, text, f)
            if m:
                frac += W_OOF * int(m.group(2)) / int(m.group(3))
        # proxy bitisinin kendi log satiri yok: sonraki fold basladiysa
        # ya da final ozet yazildiysa bitmis say
        if TRAIN_FINAL_RE.search(text) or re.search(rf"fold {f + 1}:", text):
            frac += W_PROXY
        else:
            m = _last(PROXY_PROG_RE, text, f)
            if m:
                frac += W_PROXY * int(m.group(2)) / int(m.group(3))
        total += min(frac, 1.0) / n
    return total


def _train_eta(text):
    m = _last(STEP_RE, text)
    if not m:
        return None
    fold, step, total, secs = (int(m.group(1)), int(m.group(3)),
                               int(m.group(4)), int(m.group(6)))
    if step >= total or _fold_trained(text, fold):
        return None  # egitim adimlari bitti, skorlama fazindayiz
    rate = step / max(secs, 1)
    fold_remain = (total - step) / rate
    score_pay = (total / rate) * (1 - W_TRAIN) / W_TRAIN  # OOF+proxy payi
    folds_left = N_FOLDS - fold - 1
    total_remain = fold_remain + score_pay + folds_left * (total / rate + score_pay)
    return rate, fold_remain, total_remain


def _bar(label, pct):
    """tqdm gorunumlu tek satirlik ilerleme bari (log'dan hesaplanan yuzdeyle)."""
    if tqdm is None:
        filled = int(pct / 2.5)
        return f"{label}: %{pct:5.1f} |{'#' * filled}{'.' * (40 - filled)}|"
    return tqdm.format_meter(n=pct, total=100.0, elapsed=0, ncols=56,
                             prefix=label, ascii=True,
                             bar_format="{l_bar}{bar}|")


def _staleness(path, now):
    age = now - path.stat().st_mtime
    if age > 600:
        print(f"    UYARI: log {age / 60:.0f} dk'dir guncellenmedi - "
              f"kosu durmus/takilmis olabilir")
    else:
        print(f"    log guncel ({age:.0f} sn once yazildi)")


def summarize():
    now = time.time()
    ttext = _read(TRAIN_LOG)
    if not ttext:
        print(f"{TRAIN_LOG.name} yok - egitim baslamamis")
        return

    tfinal = TRAIN_FINAL_RE.search(ttext)
    if tfinal:
        oofs = "  ".join(f"f{f}={s}" for f, s in OOF_DONE_RE.findall(ttext))
        print(_bar("EGITIM", 100.0) + "  TAMAMLANDI")
        print(f"    OOF macro_f1: {oofs}")
        print(f"    LB-proxy f1={tfinal.group(2)} (en iyi esik {tfinal.group(1)})")
    elif not STEP_RE.search(ttext) and "ornek" in ttext:
        print(_bar("EGITIM", 0.0) + "  BASLADI - isinma fazi")
        print("    ilk ilerleme satiri ~500. adimda yazilir (mdeberta'da ~10 dk);"
              " GPU kullanimiyla dogrulayin: nvidia-smi")
        _staleness(TRAIN_LOG, now)
        return
    else:
        print(_bar("EGITIM", 100 * _train_frac(ttext)) +
              "  (egitim + OOF/proxy skorlama dahil)")
        act = _last_activity(ttext)
        if act:
            print(f"    son islem: {act}")
        eta = _train_eta(ttext)
        if eta:
            rate, fold_remain, total_remain = eta
            print(f"    hiz={rate:.2f} adim/sn  bu fold ~{fold_remain / 60:.0f} dk  "
                  f"toplam ~{total_remain / 3600:.1f} saat kaldi")
        _staleness(TRAIN_LOG, now)
        return

    itext = _read(INFER_LOG)
    ifinal = INFER_FINAL_RE.search(itext) if itext else None
    if ifinal:
        print(_bar("INFERENCE", 100.0) + "  TAMAMLANDI")
        print(f"    submission: {ifinal.group(1)}")
    elif itext:
        n = N_FOLDS
        done = {int(f) for f in TEST_DONE_RE.findall(itext)}
        m = _last(TEST_PROG_RE, itext)
        cur = 0.0
        if m and int(m.group(1)) not in done:
            cur = int(m.group(2)) / int(m.group(3))
        print(_bar("INFERENCE", 100 * (len(done) + cur) / n) +
              f"  (biten fold: {len(done)}/{n})")
        act = _last_activity(itext)
        if act:
            print(f"    son islem: {act}")
        _staleness(INFER_LOG, now)
    else:
        print("INFERENCE: baslamamis  (sirada: python infer_ce.py)")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        while True:
            print("\033[2J\033[H", end="")  # ekrani temizle
            print(time.strftime("%H:%M:%S"), "-", TRAIN_LOG.parent)
            try:
                summarize()
            except Exception as e:
                print("okunamadi:", e)
            time.sleep(30)
    else:
        summarize()
