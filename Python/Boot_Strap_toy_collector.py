import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --- Problem Setup ---
# Probabilities for each of the 15 unique toys (sum to 1.0)
prob_table = np.array([.2, .1, .1, .1, .1, .1, .05, .05, .05, .05, .02, .02, .02, .02, .02])
boxes = np.arange(1, 16) # Toy IDs from 1 to 15
N_toys = len(prob_table)

def box_count(prob=prob_table, toy_ids=boxes):
    """
    Simulates the process of collecting all N_toys.
    Returns the total number of boxes opened (i).
    """
    # Use a set to efficiently track collected toys
    collected_toys = set()
    i = 0 # Box count
    
    while len(collected_toys) < N_toys:
        # 1. Sample one toy based on the probability distribution
        # np.random.choice returns an array, so we take the first element [0]
        # p=prob_table specifies the probability for each ID in toy_ids
        x = np.random.choice(toy_ids, 1, p=prob)[0]
        
        # 2. Add the toy to the collected set
        collected_toys.add(x)
        
        # 3. Increment box count
        i += 1
        
    return i

# --- Run Monte Carlo Simulation ---
TRIALS = 10000 # Increased trials for better accuracy
np.random.seed(42)
sim_boxes = np.empty(TRIALS)

for i in range(TRIALS):
    sim_boxes[i] = box_count()

# --- Calculate Results ---
est = np.mean(sim_boxes)
mcse = np.std(sim_boxes, ddof=1) / np.sqrt(TRIALS) # Monte Carlo Standard Error
confidence_level = 0.95
z_score = st.norm.ppf((1 + confidence_level) / 2) # 1.96 for 95% CI
interval = est + np.array([-1, 1]) * z_score * mcse

print(f"Number of Trials (B): {TRIALS}")
print(f"Estimated Mean Boxes (E[T]): {est:.4f}")
print(f"MC Standard Error (MCSE): {mcse:.4f}")
print(f"{confidence_level*100:.0f}% Confidence Interval: {interval}")

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
# Bins are centered on integers
bins = np.arange(sim_boxes.min(), sim_boxes.max() + 2) - 0.5 
plt.hist(sim_boxes, bins=bins, color='lightblue', edgecolor='black', density=True)
plt.axvline(est, color='red', linestyle='--', linewidth=2, label=f'Estimated Mean ({est:.2f})')
plt.title('Histogram of Total Boxes Needed (Unequal Probabilities)')
plt.xlabel('Total Boxes Opened')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()