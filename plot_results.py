
import glob
import re
import matplotlib.pyplot as plt
import os

def parse_log(filename):
    iters = []
    rewards_support = []
    rewards_query = []
    
    with open(filename, 'r') as f:
        for line in f:
            # [iter 454] loss=2.612 rew_support=-9.093 rew_query=-5.941 kl=0.0108
            match = re.search(r'\[iter (\d+)\] .* rew_support=([-\d\.]+) rew_query=([-\d\.]+)', line)
            if match:
                iters.append(int(match.group(1)))
                rewards_support.append(float(match.group(2)))
                rewards_query.append(float(match.group(3)))
    return iters, rewards_support, rewards_query

def main():
    log_files = glob.glob("results/train_log_*.txt")
    print(f"Found {len(log_files)} logs.")
    
    plt.figure(figsize=(12, 8))
    
    for log_file in log_files:
        name = os.path.basename(log_file).replace("train_log_", "").replace(".txt", "")
        iters, r_supp, r_query = parse_log(log_file)
        if not iters:
            continue
            
        plt.plot(iters, r_query, label=f"{name} (Query)", alpha=0.7)
        # plt.plot(iters, r_supp, linestyle='--', alpha=0.3) # Support clutters graph
        
    plt.xlabel("Iteration")
    plt.ylabel("Reward (Query)")
    plt.title("Mars Lander MAML Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/training_summary.png")
    print("Saved plot to results/training_summary.png")

if __name__ == "__main__":
    main()
