import numpy as np
import random
from typing import List

def generate_faithful_mock_data(n: int = 200, p_short: float = 0.4) -> List[float]:
    """
    Generates a synthetic bimodal dataset to mimic Old Faithful eruption lengths.
    It mixes two normal distributions: Short Eruptions and Long Eruptions.
    """
    n1 = int(n * p_short)
    n2 = n - n1
    
    # Component 1 (Short): Mean 2.0, StdDev 0.3
    data1 = np.random.normal(loc=2.0, scale=0.3, size=n1)
    # Component 2 (Long): Mean 4.5, StdDev 0.5
    data2 = np.random.normal(loc=4.5, scale=0.5, size=n2)
    
    eruption_data = np.concatenate([data1, data2])
    # Shuffle to mix the components naturally
    random.shuffle(eruption_data)
    
    return eruption_data.tolist()

# Generate and display the first few data points
eruption_data = generate_faithful_mock_data(n=200)
print(f"Generated {len(eruption_data)} bimodal data points.")
print(f"First 5 data points: {eruption_data[:5]}")