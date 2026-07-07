"""Tum path'ler, hiperparametreler ve seed'ler tek yerde."""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = PROJECT_DIR / "models"

# Eski projenin hazir varliklari (1.5 GB embedding'ler kopyalanmaz, path ile okunur)
OLD_PROJECT_DIR = Path(r"C:\Users\Hp\Desktop\project\trendyol_kaggle")
OLD_MODELS_DIR = OLD_PROJECT_DIR / "models"
FEATURE_BUILDER_PKL = OLD_MODELS_DIR / "feature_builder.pkl"
# Satir sirasi terms.csv / items.csv CSV sirasiyla birebir ayni
# (eski train.py: fb.fit(load_terms(...), load_items(...)))
TERM_EMB_NPY = OLD_MODELS_DIR / "feature_builder.pkl.term_emb.npy"
ITEM_EMB_NPY = OLD_MODELS_DIR / "feature_builder.pkl.item_emb.npy"

ITEMS_CSV = DATA_DIR / "items.csv"
TERMS_CSV = DATA_DIR / "terms.csv"
TRAINING_PAIRS_CSV = DATA_DIR / "training_pairs.csv"
SUBMISSION_PAIRS_CSV = DATA_DIR / "submission_pairs.csv"

SEED = 42

# ---------------------------------------------------------------- Faz 1: mining
ANN_TOP_K = 200                 # term basina saklanan en yakin item sayisi
ANN_SKIP_TOP = 5                # en yakin 5 sira hard-negatif havuzundan atlanir (false-neg korumasi)
POS_SIM_PERCENTILE = 25         # embed_sim >= termin pozitif simlerinin bu persentili ise negatif sayilmaz
HARD_PER_POS = 2                # k pozitif icin 2k hard
MEDIUM_PER_POS = 1              # k pozitif icin 1k medium (ayni yaprak kategori)
EASY_PER_POS = 1                # k pozitif icin 1k easy (rastgele)
MAX_LIST_PER_TERM = 150         # cok pozitifli termlerde liste tavani

# ---------------------------------------------------------------- Faz 1c: LB-proxy validasyon
N_PROXY_TERMS = 2000            # her seyden ayrilan term sayisi
PROXY_LIST_SIZE = 100           # test yapisini taklit eden aday listesi boyu

# ---------------------------------------------------------------- Faz 3: cross-encoder
CE_MODEL_NAME = "dbmdz/bert-base-turkish-128k-uncased"
CE_MAX_LENGTH = 128
CE_N_FOLDS = 3
CE_EPOCHS = 2
CE_LR = 2e-5
CE_WEIGHT_DECAY = 0.01
CE_WARMUP_RATIO = 0.06
CE_BATCH_SIZE = 32
CE_GRAD_ACCUM = 2
CE_INFER_BATCH_SIZE = 256
HF_CACHE = str(MODELS_DIR / "hf_cache")

for _d in (ARTIFACTS_DIR, OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
