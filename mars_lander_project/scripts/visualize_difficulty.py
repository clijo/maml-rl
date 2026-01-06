
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from maml_rl.envs.mars_lander import MarsLanderEnv

def verify_difficulty():
    difficulties = [0.0, 0.5, 1.0]
    num_samples = 50
    
    plt.figure(figsize=(15, 5))
    
    for i, diff in enumerate(difficulties):
        tasks = MarsLanderEnv.sample_tasks(num_samples, difficulty=diff)
        
        # Plot connections
        for sx, sy, tx, lw in zip(start_xs, start_ys, target_xs, landing_widths):
            plt.plot([sx, tx], [sy, 0], 'k-', alpha=0.1)
            # Plot landing zone
            plt.plot([tx - lw/2, tx + lw/2], [0, 0], 'g-', linewidth=2, alpha=0.5)
            
        plt.scatter(start_xs, start_ys, c=gravities, cmap='viridis', label='Start (Color=Gravity)', alpha=0.7)
        plt.scatter(target_xs, np.zeros_like(target_xs), c='r', marker='x', label='Target Center')
        
        plt.xlim(0, 7000)
        plt.ylim(-100, 3000)
        plt.xlabel("X Position")
        plt.ylabel("Altitude")
        
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig('mars_lander_project/plots/difficulty_visualization.png')
    print("Saved mars_lander_project/plots/difficulty_visualization.png")

if __name__ == "__main__":
    try:
        verify_difficulty()
    except Exception as e:
        print(f"Error: {e}")
