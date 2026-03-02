#!/usr/bin/env python3
"""
Quick script to calculate training progress for DeComFL
"""
import sys

def calculate_progress(eval_round: int, total_iterations: int = 100, eval_iterations: int = 20):
    """
    Calculate how many iterations have been completed and how many remain.
    
    Args:
        eval_round: Current evaluation round number
        total_iterations: Total training iterations (default: 100)
        eval_iterations: Evaluate every N iterations (default: 20)
    """
    # Evaluation happens at iterations: eval_iterations, 2*eval_iterations, 3*eval_iterations, ...
    # So evaluation round N happens at iteration: N * eval_iterations
    
    current_iteration = eval_round * eval_iterations
    remaining_iterations = total_iterations - current_iteration
    remaining_eval_rounds = (total_iterations // eval_iterations) - eval_round
    
    print(f"📊 Training Progress Calculation")
    print(f"{'='*50}")
    print(f"Current Evaluation Round: {eval_round}")
    print(f"Total Iterations: {total_iterations}")
    print(f"Evaluation Frequency: Every {eval_iterations} iterations")
    print(f"{'='*50}")
    print(f"Current Iteration: ~{current_iteration}")
    print(f"Remaining Iterations: {remaining_iterations}")
    print(f"Remaining Evaluation Rounds: {remaining_eval_rounds}")
    print(f"{'='*50}")
    
    if current_iteration >= total_iterations:
        print("✅ Training should be complete!")
    elif remaining_iterations > 0:
        print(f"⏳ Still running... {remaining_iterations} iterations remaining")
    
    # Show all evaluation points
    eval_points = [i * eval_iterations for i in range(1, (total_iterations // eval_iterations) + 1)]
    if total_iterations % eval_iterations == 0:
        eval_points.append(total_iterations)
    print(f"\nAll Evaluation Points: {eval_points}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_training_progress.py <eval_round> [total_iterations] [eval_iterations]")
        print("\nExample:")
        print("  python check_training_progress.py 16 100 20")
        print("  python check_training_progress.py 16 100 25")
        print("\nDefaults: total_iterations=100, eval_iterations=20")
        sys.exit(1)
    
    eval_round = int(sys.argv[1])
    total_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    eval_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    calculate_progress(eval_round, total_iterations, eval_iterations)
