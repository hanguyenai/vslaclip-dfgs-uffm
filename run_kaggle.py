"""
Run all configs for 47 submissions budget (36 configs + 11 reserve)
Fixed: cffm_k = 12

Now supports:
- --start N        : start from config index N (0-based), e.g. --start 12 to start at c12
- --submit-only    : only submit existing CSVs, don't run evaluation
"""

import subprocess
import time
import os
import itertools
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

CONFIG_FILE = "configs/adapter/vit_adapter.yml"
MODEL_PATH = "output_original/VSLACLIPv2/ViT-B-16_stage3_final.pth"  # <-- THAY ĐỔI ĐƯỜNG DẪN MODEL
COMPETITION = "detreidxv1"
OUTPUT_DIR = "output"
DELAY_BETWEEN_SUBMIT = 3  # seconds

# Fixed param
CFFM_K = 12

# ============================================
# PARAMETER GRID (36 configs)
# ============================================

# Fusion methods
FUSIONS = ["weighted_mean", "self_and_neighbors", "mean"]

# Temperatures (log scale)
TEMPERATURES = [0.03, 0.05, 0.1, 0.2, 0.5]

# AMC weight combinations (alpha, beta, gamma) - will be normalized
AMC_WEIGHTS = [
    (0.5, 0.4, 0.1),   # Direct heavy
    (0.4, 0.5, 0.1),   # URF heavy
    (0.4, 0.4, 0.2),   # Balanced with CCE
    (0.5, 0.3, 0.2),   # Direct + CCE
    (0.3, 0.5, 0.2),   # URF + CCE
    (0.35, 0.35, 0.3), # High CCE
    (0.45, 0.45, 0.1), # Low CCE
    (0.6, 0.3, 0.1),   # Very direct heavy
]

# ============================================
# GENERATE ALL CONFIGS
# ============================================

def generate_configs():
    """Generate all config combinations."""
    configs = []
    idx = 0
    
    # Main grid: fusion × temperature × selected AMC weights
    for fusion in FUSIONS:
        for temp in TEMPERATURES:
            # Select 2-3 AMC weights per fusion-temp combo
            if fusion == "weighted_mean":
                # Most promising, test more AMC combos
                amc_subset = AMC_WEIGHTS[:5]
            elif fusion == "self_and_neighbors":
                amc_subset = AMC_WEIGHTS[:4]
            else:  # mean
                amc_subset = AMC_WEIGHTS[:3]
            
            for alpha, beta, gamma in amc_subset:
                # Normalize
                total = alpha + beta + gamma
                a, b, g = alpha/total, beta/total, gamma/total
                
                configs.append({
                    "id": f"c{idx:02d}",
                    "cffm_k": CFFM_K,
                    "cffm_fusion": fusion,
                    "cffm_temperature": temp,
                    "alpha": round(a, 3),
                    "beta": round(b, 3),
                    "gamma": round(g, 3),
                })
                idx += 1
                
                if idx >= 36:  # Budget limit
                    return configs
    
    return configs


def generate_configs_simple():
    """Simpler grid - exactly 36 configs."""
    configs = []
    idx = 0
    
    # 3 fusions × 4 temps × 3 AMC = 36 configs exactly
    fusions = ["weighted_mean", "self_and_neighbors", "mean"]
    temps = [0.05, 0.1, 0.2, 0.5]
    amcs = [
        (0.5, 0.4, 0.1),   # Direct heavy
        (0.4, 0.5, 0.1),   # URF heavy  
        (0.4, 0.4, 0.2),   # Balanced
    ]
    
    for fusion in fusions:
        for temp in temps:
            for alpha, beta, gamma in amcs:
                total = alpha + beta + gamma
                configs.append({
                    "id": f"c{idx:02d}",
                    "cffm_k": CFFM_K,
                    "cffm_fusion": fusion,
                    "cffm_temperature": temp,
                    "alpha": round(alpha/total, 3),
                    "beta": round(beta/total, 3),
                    "gamma": round(gamma/total, 3),
                })
                idx += 1
    
    return configs


# ============================================
# RUN & SUBMIT
# ============================================

def run_evaluation(config):
    """Run evaluation for a config."""
    
    cid = config["id"]
    
    cmd = [
        "python", "evaluate_all_cases_amc.py",
        "--config_file", CONFIG_FILE,
        "--model_path", MODEL_PATH,
        "--cases", "1,2,3",
        "--cffm_k", str(config["cffm_k"]),
        "--cffm_fusion", config["cffm_fusion"],
        "--cffm_temperature", str(config["cffm_temperature"]),
        "--alpha", str(config["alpha"]),
        "--beta", str(config["beta"]),
        "--gamma", str(config["gamma"]),
    ]
    
    print(f"\n[{cid}] Running: k={config['cffm_k']}, {config['cffm_fusion']}, "
          f"t={config['cffm_temperature']}, "
          f"α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Rename output file
        default_csv = f"{OUTPUT_DIR}/evaluation_rankings_amc_epoch60.csv"
        new_csv = f"{OUTPUT_DIR}/{cid}.csv"
        
        if os.path.exists(default_csv):
            os.rename(default_csv, new_csv)
            print(f"    ✓ Saved: {new_csv}")
            return new_csv
        else:
            print(f"    ✗ Output not found: {default_csv}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error: {e.stderr[:200] if e.stderr else e}")
        return None


def submit_to_kaggle(config, csv_path):
    """Submit to Kaggle."""
    
    cid = config["id"]
    
    # Short message for tracking
    msg = (f"{cid}_k{config['cffm_k']}_"
           f"{config['cffm_fusion'][:2]}_"
           f"t{config['cffm_temperature']}_"
           f"a{config['alpha']}b{config['beta']}g{config['gamma']}")
    
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION,
        "-f", csv_path,
        "-m", msg
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"    ✓ Submitted: {msg}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Submit failed: {e.stderr[:200] if e.stderr else e}")
        return False


def run_all(auto_submit=True, use_simple_grid=True, start_idx=0):
    """
    Run all configs from start_idx (0-based).
    Example: start_idx=12 -> start at c12.
    """
    
    # Generate configs
    if use_simple_grid:
        configs = generate_configs_simple()
    else:
        configs = generate_configs()
    
    total_configs = len(configs)
    if start_idx < 0 or start_idx >= total_configs:
        raise ValueError(f"start_idx {start_idx} out of range [0, {total_configs-1}]")
    
    print("=" * 60)
    print(f"TOTAL CONFIGS:  {total_configs}")
    print(f"START INDEX:    {start_idx} (id={configs[start_idx]['id']})")
    print(f"AUTO SUBMIT:    {auto_submit}")
    print(f"COMPETITION:    {COMPETITION}")
    print(f"MODEL:          {MODEL_PATH}")
    print("=" * 60)
    
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Track results
    results = []
    start_time = time.time()
    
    # Only run from start_idx onwards
    to_run = configs[start_idx:]
    num_to_run = len(to_run)
    
    for i, config in enumerate(to_run, start=start_idx):
        print(f"\n{'='*60}")
        print(f"Progress: {i - start_idx + 1}/{num_to_run}  (global index {i}/{total_configs-1})")
        print(f"{'='*60}")
        
        # Run evaluation
        csv_path = run_evaluation(config)
        
        result = {
            "id": config["id"],
            "config": config,
            "csv": csv_path,
            "submitted": False
        }
        
        # Submit if successful
        if csv_path and auto_submit:
            time.sleep(1)
            result["submitted"] = submit_to_kaggle(config, csv_path)
            
            # only wait if there are more configs to run
            if i < total_configs - 1:
                print(f"    Waiting {DELAY_BETWEEN_SUBMIT}s...")
                time.sleep(DELAY_BETWEEN_SUBMIT)
        
        results.append(result)
    
    # Summary (for this run)
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r["csv"])
    submitted = sum(1 for r in results if r["submitted"])
    
    print("\n" + "=" * 60)
    print("SUMMARY (THIS RUN)")
    print("=" * 60)
    print(f"Total configs:       {total_configs}")
    print(f"Start index:         {start_idx}")
    print(f"Ran configs:         {num_to_run}")
    print(f"Successful runs:     {successful}")
    print(f"Submitted:           {submitted}")
    print(f"Total time:          {elapsed/60:.1f} minutes")
    print(f"Remaining subs (47): {47 - submitted}")
    
    # Save results log
    log_file = f"{OUTPUT_DIR}/run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write(f"Run completed at {datetime.now()}\n")
        f.write(f"Total configs: {total_configs}, Start index: {start_idx}, "
                f"Ran: {num_to_run}, Submitted: {submitted}\n\n")
        for r in results:
            c = r["config"]
            f.write(f"{r['id']}: {c['cffm_fusion']}, t={c['cffm_temperature']}, "
                    f"α={c['alpha']}, β={c['beta']}, γ={c['gamma']} "
                    f"-> {'✓' if r['submitted'] else '✗'}\n")
    
    print(f"\nLog saved: {log_file}")
    
    return results


def submit_only(use_simple_grid=True, start_idx=0):
    """
    Only submit existing CSVs for configs from start_idx.
    Không chạy lại evaluate_all_cases_amc.py.
    """
    # Generate configs
    if use_simple_grid:
        configs = generate_configs_simple()
    else:
        configs = generate_configs()
    
    total_configs = len(configs)
    if start_idx < 0 or start_idx >= total_configs:
        raise ValueError(f"start_idx {start_idx} out of range [0, {total_configs-1}]")
    
    print("=" * 60)
    print("SUBMIT ONLY MODE")
    print("=" * 60)
    print(f"TOTAL CONFIGS:  {total_configs}")
    print(f"START INDEX:    {start_idx} (id={configs[start_idx]['id']})")
    print(f"COMPETITION:    {COMPETITION}")
    print("=" * 60)
    
    submitted = 0
    missing = 0
    failed = 0
    
    to_run = configs[start_idx:]
    num_to_run = len(to_run)
    
    for i, config in enumerate(to_run, start=start_idx):
        cid = config["id"]
        csv_path = os.path.join(OUTPUT_DIR, f"{cid}.csv")
        
        print(f"\n{'='*60}")
        print(f"[SUBMIT ONLY] Progress: {i - start_idx + 1}/{num_to_run}  (global index {i}/{total_configs-1})")
        print(f"Config: {cid}  fusion={config['cffm_fusion']}, t={config['cffm_temperature']}, "
              f"α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")
        
        if not os.path.exists(csv_path):
            print(f"    ✗ CSV not found: {csv_path}")
            missing += 1
            continue
        
        ok = submit_to_kaggle(config, csv_path)
        if ok:
            submitted += 1
        else:
            failed += 1
        
        # delay giữa các submit
        if i < total_configs - 1:
            print(f"    Waiting {DELAY_BETWEEN_SUBMIT}s...")
            time.sleep(DELAY_BETWEEN_SUBMIT)
    
    print("\n" + "=" * 60)
    print("SUMMARY (SUBMIT ONLY)")
    print("=" * 60)
    print(f"Total configs:       {total_configs}")
    print(f"Start index:         {start_idx}")
    print(f"Checked configs:     {num_to_run}")
    print(f"Submitted:           {submitted}")
    print(f"Missing CSV:         {missing}")
    print(f"Submit failed:       {failed}")
    print(f"Remaining subs (47): {47 - submitted}")
    print("=" * 60)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH, help="Path to model")
    parser.add_argument("--no-submit", action="store_true", help="Don't submit to Kaggle (for run_all)")
    parser.add_argument("--full-grid", action="store_true", help="Use full grid instead of simple")
    parser.add_argument("--dry-run", action="store_true", help="Just print configs, don't run")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) in config list, e.g. 12 to start at c12",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Only submit existing CSVs, don't run evaluation",
    )
    
    args = parser.parse_args()
    
    # Update model path
    MODEL_PATH = args.model_path
    
    # MODE 1: submit-only
    if args.submit_only:
        # Nếu muốn, có thể kết hợp với --dry-run (in ra list sẽ submit, không gửi thật)
        if args.dry_run:
            configs = generate_configs_simple() if not args.full_grid else generate_configs()
            total = len(configs)
            if args.start < 0 or args.start >= total:
                raise ValueError(f"--start {args.start} out of range [0, {total-1}]")
            print(f"[DRY RUN - SUBMIT ONLY] Would submit {total - args.start} configs (from index {args.start} to {total-1}):\n")
            for c in configs[args.start:]:
                cid = c["id"]
                csv_path = os.path.join(OUTPUT_DIR, f"{cid}.csv")
                print(f"  {cid}: csv={csv_path}, fusion={c['cffm_fusion']}, "
                      f"temp={c['cffm_temperature']}, α={c['alpha']}, β={c['beta']}, γ={c['gamma']}")
        else:
            submit_only(
                use_simple_grid=not args.full_grid,
                start_idx=args.start,
            )
    
    # MODE 2: dry-run (chỉ in config, không chạy, không submit)
    elif args.dry_run:
        configs = generate_configs_simple() if not args.full_grid else generate_configs()
        total = len(configs)
        
        if args.start < 0 or args.start >= total:
            raise ValueError(f"--start {args.start} out of range [0, {total-1}]")
        
        print(f"Would run {total - args.start} configs (from index {args.start} to {total-1}):\n")
        for c in configs[args.start:]:
            print(f"  {c['id']}: fusion={c['cffm_fusion']}, temp={c['cffm_temperature']}, "
                  f"α={c['alpha']}, β={c['beta']}, γ={c['gamma']}")
    
    # MODE 3: bình thường – chạy eval (+ optional submit)
    else:
        run_all(
            auto_submit=not args.no_submit,
            use_simple_grid=not args.full_grid,
            start_idx=args.start,
        )
