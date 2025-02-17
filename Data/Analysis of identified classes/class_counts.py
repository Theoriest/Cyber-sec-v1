import pandas as pd

def count_cia_classes(file_paths):
    results = {}
    for file_path in file_paths:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Check if 'CIA_Class' column exists
        if 'CIA_Class' in df.columns:
            # Count the occurrences of each class in the 'CIA_Class' column
            class_counts = df['CIA_Class'].value_counts()
            results[file_path] = class_counts
        else:
            results[file_path] = "CIA_Class column not found"
    
    return results

# Example usage
file_paths = [
    "/workspace/Cyber-sec-v1/Data/CSV Files/CVSS_without_tf.csv",
    "/workspace/Cyber-sec-v1/Data/CSV Files/CVSS_with_tf.csv",
    "/workspace/Cyber-sec-v1/Data/CSV Files/cve_data.csv/CVSS_data_pre-trained.csv"
]

class_counts = count_cia_classes(file_paths)
for file_path, counts in class_counts.items():
    print(f"Class counts for {file_path}:")
    print(counts)
    print()