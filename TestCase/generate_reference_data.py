import numpy as np
import csv

def generate_reference_data(filename="reference_data.csv", n_points=5000):
    """
    Generates reference data for a simple spiraling decay problem.
    Inputs: u, v
    Output: y
    """
    t = np.linspace(0, 1, n_points)
    
    # Spiraling inputs with decaying amplitude
    u = (1 - t) * np.cos(48 * np.pi * t)
    v = (1 - t) * np.sin(48 * np.pi * t)
    
    # Non-linear output
    y = np.sin(2 * np.pi * (u**2 + v))
    
    X_dim = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    Y_dim = y[:, np.newaxis]

    # Save to CSV
    with open(filename, "w+") as fid:
        fid.write("u\tv\ty\n")
        csvWriter = csv.writer(fid, delimiter="\t")
        csvWriter.writerows(np.hstack((X_dim, Y_dim)))
        
    print(f"Successfully generated {n_points} data points in '{filename}'")

if __name__ == "__main__":
    generate_reference_data()