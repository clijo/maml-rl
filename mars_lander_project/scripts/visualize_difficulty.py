
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from maml_rl.envs.mars_lander import MarsLanderEnv

def verify_difficulty():
    difficulties = [0.0, 0.5, 1.0]
    num_samples = 50
    
    plt.figure(figsize=(15, 5))
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, diff in enumerate(difficulties):
        # Sample just ONE task to show as a clear example
        task = MarsLanderEnv.sample_tasks(1, difficulty=diff)[0]
        
        ax = axes[i]
        
        # Extract params
        start_x = task['start_x']
        start_y = task['start_y']
        target_x = task['target_x']
        landing_width = task.get('landing_width', 1000.0)
        gravity = task['gravity']
        wind = task['wind']
        
        # Initial conditions (often randomized in reset, but controlled in Easy mode)
        # We'll simulate what reset does or show what's in the task if we put it there
        start_hs = task.get('start_hs', 'Random') 
        start_rotate = task.get('start_rotate', 'Random')
        
        if isinstance(start_hs, float):
            hs_str = f"{start_hs:.1f} m/s"
        else:
            hs_str = "Random (-50, 50)"

        if isinstance(start_rotate, float):
            rot_str = f"{start_rotate:.1f} deg"
        else:
            rot_str = "Random (-90, 90)"
            
        # Plot Landscape (Ground)
        ax.plot([0, 7000], [0, 0], 'k-', lw=1)
        
        # Plot Landing Zone
        ax.plot([target_x - landing_width/2, target_x + landing_width/2], [0, 0], 'g-', lw=4, label='Landing Zone')
        
        # Plot Start Position
        ax.scatter(start_x, start_y, c='blue', s=100, label='Start', zorder=5)
        
        # Draw Arrow for Gravity (Visual aid)
        # Length proportional to gravity?
        ax.arrow(6500, 2500, 0, -gravity*100, head_width=100, head_length=100, fc='purple', ec='purple', label='Gravity')

        # Draw Wind
        ax.arrow(6500, 2500, wind[0]*100, wind[1]*100, head_width=100, head_length=100, fc='cyan', ec='cyan', label='Wind')
        
        # Annotation Box
        text_str = (f"Difficulty: {diff}\n"
                    f"Gravity: {gravity:.2f} m/s²\n"
                    f"Wind: ({wind[0]:.1f}, {wind[1]:.1f}) m/s\n"
                    f"Start HS: {hs_str}\n"
                    f"Start Rot: {rot_str}\n"
                    f"Land Width: {landing_width:.0f}m")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.set_title(f"Scenario Example (Diff {diff})")
        ax.set_xlim(0, 7000)
        ax.set_ylim(-100, 3000)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='lower right')
            
    plt.tight_layout()
    plt.savefig('mars_lander_project/plots/difficulty_visualization.png')
    print("Saved mars_lander_project/plots/difficulty_visualization.png")

if __name__ == "__main__":
    try:
        verify_difficulty()
    except Exception as e:
        print(f"Error: {e}")
