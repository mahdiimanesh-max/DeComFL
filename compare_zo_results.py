#!/usr/bin/env python3
"""
Compare ZO RGE results: With FL vs Without FL
Extracts evaluation metrics from log files and creates a comparison report.
"""
import re
import sys
from pathlib import Path
from typing import Optional


def extract_eval_metrics(log_file: Path) -> list[dict]:
    """Extract evaluation metrics from log file."""
    metrics = []
    
    if not log_file.exists():
        print(f"⚠️  Warning: {log_file} not found")
        return metrics
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern for zo_rge_main.py: "Evaluation(round {epoch}): Eval Loss:{loss:.4f}, Accuracy:{acc:.2f}%"
    pattern1 = r'Evaluation\(round (\d+)\): Eval Loss:([\d.]+), Accuracy:([\d.]+)%'
    matches1 = re.findall(pattern1, content)
    
    # Pattern for decomfl_main.py: "Evaluation(Iteration {ite}): Eval Loss:{loss:.4f}, Accuracy:{acc:.2f}%"
    pattern2 = r'Evaluation\(Iteration (\d+)\):.*?Eval Loss:([\d.]+), Accuracy:([\d.]+)%'
    matches2 = re.findall(pattern2, content)
    
    matches = matches1 + matches2
    
    for match in matches:
        epoch_or_iter = int(match[0])
        loss = float(match[1])
        accuracy = float(match[2])
        metrics.append({
            'epoch_or_iter': epoch_or_iter,
            'loss': loss,
            'accuracy': accuracy
        })
    
    return sorted(metrics, key=lambda x: x['epoch_or_iter'])


def print_comparison(no_fl_metrics: list[dict], with_fl_metrics: list[dict], results_dir: Path):
    """Print and save comparison report."""
    
    print("\n" + "="*80)
    print("📊 COMPARISON REPORT: ZO RGE With FL vs Without FL")
    print("="*80)
    
    # Create comparison table
    print("\n┌─────────────┬──────────────────────┬──────────────────────┬──────────────┐")
    print("│   Round     │   Without FL         │   With FL            │   Difference │")
    print("│             │   Loss    Accuracy   │   Loss    Accuracy   │   Accuracy  │")
    print("├─────────────┼──────────────────────┼──────────────────────┼──────────────┤")
    
    max_rounds = max(len(no_fl_metrics), len(with_fl_metrics))
    comparison_data = []
    
    for i in range(max_rounds):
        no_fl = no_fl_metrics[i] if i < len(no_fl_metrics) else None
        with_fl = with_fl_metrics[i] if i < len(with_fl_metrics) else None
        
        if no_fl and with_fl:
            round_num = no_fl['epoch_or_iter']
            no_fl_loss = no_fl['loss']
            no_fl_acc = no_fl['accuracy']
            with_fl_loss = with_fl['loss']
            with_fl_acc = with_fl['accuracy']
            diff = with_fl_acc - no_fl_acc
            
            comparison_data.append({
                'round': round_num,
                'no_fl_loss': no_fl_loss,
                'no_fl_acc': no_fl_acc,
                'with_fl_loss': with_fl_loss,
                'with_fl_acc': with_fl_acc,
                'diff': diff
            })
            
            print(f"│ {round_num:11d} │ {no_fl_loss:6.4f}  {no_fl_acc:6.2f}%   │ "
                  f"{with_fl_loss:6.4f}  {with_fl_acc:6.2f}%   │ {diff:+7.2f}%  │")
        elif no_fl:
            print(f"│ {no_fl['epoch_or_iter']:11d} │ {no_fl['loss']:6.4f}  {no_fl['accuracy']:6.2f}%   │ "
                  f"{'N/A':>20} │ {'N/A':>12} │")
        elif with_fl:
            print(f"│ {with_fl['epoch_or_iter']:11d} │ {'N/A':>20} │ "
                  f"{with_fl['loss']:6.4f}  {with_fl['accuracy']:6.2f}%   │ {'N/A':>12} │")
    
    print("└─────────────┴──────────────────────┴──────────────────────┴──────────────┘")
    
    # Summary statistics
    if comparison_data:
        print("\n" + "="*80)
        print("📈 SUMMARY STATISTICS")
        print("="*80)
        
        final_no_fl = no_fl_metrics[-1] if no_fl_metrics else None
        final_with_fl = with_fl_metrics[-1] if with_fl_metrics else None
        
        if final_no_fl and final_with_fl:
            print(f"\n🔹 Final Results (Round {final_no_fl['epoch_or_iter']}):")
            print(f"   Without FL: Loss={final_no_fl['loss']:.4f}, Accuracy={final_no_fl['accuracy']:.2f}%")
            print(f"   With FL:    Loss={final_with_fl['loss']:.4f}, Accuracy={final_with_fl['accuracy']:.2f}%")
            print(f"   Difference: Accuracy={final_with_fl['accuracy'] - final_no_fl['accuracy']:+.2f}%")
        
        # Average accuracy
        avg_no_fl = sum(m['accuracy'] for m in no_fl_metrics) / len(no_fl_metrics) if no_fl_metrics else 0
        avg_with_fl = sum(m['accuracy'] for m in with_fl_metrics) / len(with_fl_metrics) if with_fl_metrics else 0
        
        print(f"\n🔹 Average Accuracy Across All Rounds:")
        print(f"   Without FL: {avg_no_fl:.2f}%")
        print(f"   With FL:    {avg_with_fl:.2f}%")
        print(f"   Difference: {avg_with_fl - avg_no_fl:+.2f}%")
        
        # Best accuracy
        best_no_fl = max(no_fl_metrics, key=lambda x: x['accuracy']) if no_fl_metrics else None
        best_with_fl = max(with_fl_metrics, key=lambda x: x['accuracy']) if with_fl_metrics else None
        
        if best_no_fl and best_with_fl:
            print(f"\n🔹 Best Accuracy:")
            print(f"   Without FL: {best_no_fl['accuracy']:.2f}% (Round {best_no_fl['epoch_or_iter']})")
            print(f"   With FL:    {best_with_fl['accuracy']:.2f}% (Round {best_with_fl['epoch_or_iter']})")
    
    # Save comparison to file
    report_file = results_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("ZO RGE Comparison: With FL vs Without FL\n")
        f.write("="*80 + "\n\n")
        for data in comparison_data:
            f.write(f"Round {data['round']}: "
                   f"NoFL(Loss={data['no_fl_loss']:.4f}, Acc={data['no_fl_acc']:.2f}%) | "
                   f"FL(Loss={data['with_fl_loss']:.4f}, Acc={data['with_fl_acc']:.2f}%) | "
                   f"Diff={data['diff']:+.2f}%\n")
    
    print(f"\n✅ Comparison report saved to: {report_file}")
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_zo_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    no_fl_log = results_dir / "zo_rge_no_fl.log"
    with_fl_log = results_dir / "zo_rge_with_fl.log"
    
    print(f"📂 Reading results from: {results_dir}")
    print()
    
    # Extract metrics
    no_fl_metrics = extract_eval_metrics(no_fl_log)
    with_fl_metrics = extract_eval_metrics(with_fl_log)
    
    print(f"📊 Found {len(no_fl_metrics)} evaluation rounds (Without FL)")
    print(f"📊 Found {len(with_fl_metrics)} evaluation rounds (With FL)")
    
    if not no_fl_metrics and not with_fl_metrics:
        print("⚠️  No evaluation metrics found in log files!")
        return
    
    # Print comparison
    print_comparison(no_fl_metrics, with_fl_metrics, results_dir)


if __name__ == "__main__":
    main()
