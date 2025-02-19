import requests
import pandas as pd
import time
import os

# API endpoint for CVE data (NVD API v2.0)
base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Parameters for pagination
results_per_page = 2000

# Update the path to a directory within your home directory
csv_file_path = os.path.expanduser("/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/cve_data.csv")

# Ensure the directory exists
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# Function to fetch CVE data and return as a list of dictionaries
def fetch_cve_data(start_index, results_per_page):
    url = f"{base_url}?startIndex={start_index}&resultsPerPage={results_per_page}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        vulnerabilities = data.get("vulnerabilities", [])
        total_results = data.get("totalResults", 0)
        
        print(f"Fetched {len(vulnerabilities)} records, Total available: {total_results}")
        
        cve_data = []
        
        for vuln in vulnerabilities:
            cve = vuln.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            description_text = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description_text = desc.get("value", "")
                    break
            
            metrics = cve.get("metrics", {})
            cvss_score = None
            if "cvssMetricV31" in metrics:
                cvss_list = metrics.get("cvssMetricV31", [])
                if cvss_list:
                    cvss_data = cvss_list[0].get("cvssData", {})
                    cvss_score = cvss_data.get("baseScore", None)
            
            published_date = cve.get("published", "")
            references = cve.get("references", {})
            reference_urls = [ref.get("url", "") for ref in references]
            
            cve_data.append({
                "CVE_ID": cve_id,
                "Description": description_text,
                "CVSS_Score": cvss_score,
                "Published": published_date,
                "References": reference_urls
            })
        
        return cve_data, total_results
    else:
        print(f"API request failed with status code {response.status_code}")
        time.sleep(60)
        return [], 0

# Determine the starting index based on the existing CSV file
if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    existing_df = pd.read_csv(csv_file_path)
    start_index = len(existing_df) + 1  # Start from the next record
else:
    start_index = 0

filtered_data = []

# Fetch and save CVE data
while True:
    cve_data, total_results = fetch_cve_data(start_index, results_per_page)
    
    if not cve_data:
        break
    
    filtered_data.extend(cve_data)
    start_index += results_per_page
    
    if start_index >= total_results:
        break

# Convert the filtered data into a Pandas DataFrame
df = pd.DataFrame(filtered_data)

# Save DataFrame to CSV file
if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_file_path, mode='w', header=True, index=False)

print(f"Data saved to {csv_file_path}")
print(f"Total records fetched: {len(df)}")