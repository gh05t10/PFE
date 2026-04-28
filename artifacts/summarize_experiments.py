import json
from pathlib import Path

base_dir = Path("/home/peter/PFE/processed/chl_shallow/")

summaries = []

# GRU summaries
for p in base_dir.rglob("gru_eval_summary.json"):
    try:
        data = json.loads(p.read_text())
        window = p.parent.parent.name
        resample = p.parent.parent.parent.parent.name
        
        # handle different metric key names
        v_rmse = data.get("val_rmse_chl", data.get("val_rmse", None))
        t_rmse = data.get("test_rmse_chl", data.get("test_rmse", None))
        
        summaries.append({
            "model": "GRU",
            "resample": resample,
            "window": window,
            "val_rmse": v_rmse,
            "test_rmse": t_rmse
        })
    except Exception:
        pass

# SLIDE summaries
for p in base_dir.rglob("slide_eval_summary.json"):
    try:
        data = json.loads(p.read_text())
        window = p.parent.parent.name
        resample = p.parent.parent.parent.parent.name
        
        v_rmse = data.get("val_rmse_chl", data.get("val_rmse", None))
        t_rmse = data.get("test_rmse_chl", data.get("test_rmse", None))

        model_name = "SLIDE"
        if "student" in p.parent.name:
            model_name = "SLIDE_Student"
            
        summaries.append({
            "model": model_name,
            "resample": resample,
            "window": window,
            "val_rmse": v_rmse,
            "test_rmse": t_rmse
        })
    except Exception:
        pass

print(f"{'Model':<15} | {'Resample':<20} | {'Window Config':<30} | {'Val RMSE':<10} | {'Test RMSE':<10}")
print("-" * 95)
summaries.sort(key=lambda x: (x["resample"], x["window"], x["model"]))
for s in summaries:
    v = f"{s['val_rmse']:.4f}" if s['val_rmse'] is not None else "N/A"
    t = f"{s['test_rmse']:.4f}" if s['test_rmse'] is not None else "N/A"
    print(f"{s['model']:<15} | {s['resample']:<20} | {s['window']:<30} | {v:<10} | {t:<10}")
