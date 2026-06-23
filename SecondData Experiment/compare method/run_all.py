#!/usr/bin/env python3
"""Run all comparison methods on HAM10000 and generate comparison table.

Usage:
    python run_all.py --epochs 50
    python run_all.py --methods resnet50 densenet121  # run specific methods only
"""
import os, sys, json, argparse, time, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = SCRIPT_DIR

ALL_METHODS = ['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
               'dinov2_linear', 'rulefit', 'CBM', 'CRL']


def run_method(method, epochs, gpu):
    """Run a single method and return results."""
    print(f"\n{'='*70}", flush=True)
    print(f"Running method: {method}", flush=True)
    print(f"{'='*70}", flush=True)

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, 'train.py'),
        '--method', method,
        '--epochs', str(epochs),
        '--gpu', str(gpu),
    ]

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)

    # Read results
    results_path = os.path.join(OUTPUT_BASE, method, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def generate_comparison_table(all_results):
    """Generate a formatted comparison table."""
    print(f"\n\n{'='*90}", flush=True)
    print(f"{'HAM10000 Comparison Results':^90}", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Method':<25} {'Accuracy':>10} {'Balanced Acc':>14} {'AUC':>10} {'Time (min)':>12}", flush=True)
    print(f"{'-'*90}", flush=True)

    for method in ALL_METHODS:
        if method in all_results and all_results[method]:
            r = all_results[method]
            print(f"{method:<25} {r['acc']:>10.4f} {r['bal_acc']:>14.4f} {r['auc']:>10.4f} {r['time_seconds']/60:>12.1f}", flush=True)
        else:
            print(f"{method:<25} {'N/A':>10} {'N/A':>14} {'N/A':>10} {'N/A':>12}", flush=True)

    print(f"{'='*90}", flush=True)

    # Find best
    valid = {k: v for k, v in all_results.items() if v}
    if valid:
        best_acc = max(valid.values(), key=lambda x: x['acc'])
        best_bal = max(valid.values(), key=lambda x: x['bal_acc'])
        best_auc = max(valid.values(), key=lambda x: x['auc'])
        print(f"\nBest Accuracy:      {best_acc['method']} ({best_acc['acc']:.4f})", flush=True)
        print(f"Best Balanced Acc:  {best_bal['method']} ({best_bal['bal_acc']:.4f})", flush=True)
        print(f"Best AUC:           {best_auc['method']} ({best_auc['auc']:.4f})", flush=True)

    # Save table
    table_path = os.path.join(OUTPUT_BASE, 'comparison_results.json')
    with open(table_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {table_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run all HAM10000 comparison methods')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        help='Methods to run (default: all)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    all_results = {}

    for method in args.methods:
        if method not in ALL_METHODS:
            print(f"Unknown method: {method}, skipping", flush=True)
            continue

        # Check if results already exist
        results_path = os.path.join(OUTPUT_BASE, method, 'results.json')
        if os.path.exists(results_path):
            print(f"\n{method}: results already exist, loading...", flush=True)
            with open(results_path) as f:
                all_results[method] = json.load(f)
            continue

        result = run_method(method, args.epochs, args.gpu)
        all_results[method] = result

    generate_comparison_table(all_results)


if __name__ == '__main__':
    main()
