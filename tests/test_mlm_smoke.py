import os
import subprocess


def test_mlm_smoke_run(tmp_path):
    # run training script in fast mlm mode; use PYTHONPATH to import src/
    # ensure data artifact exists in CI by running the fast prepare step
    import os
    os.makedirs("data", exist_ok=True)
    prep = ["python", "scripts/prepare_pbmc3k.py", "--fast", "--out", "data/pbmc3k_hvg_knn.npz"]
    rprep = subprocess.run(prep, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(rprep.stdout)

    cmd = ["python", "scripts/train_classifier.py", "--fast", "--mlm", "--config", "configs/default.yaml"]
    env = os.environ.copy()
    env["PYTHONPATH"] = "./src"
    # run and ensure exit code 0
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    assert r.returncode == 0
    # artifacts should be present
    assert os.path.exists("data/scrna_longformer_cls.pt")
    assert os.path.exists("data/pbmc3k_emb_cls.npy")
