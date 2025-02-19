import pandas as pd

def clean_cve_data(df):
    # Remove blank descriptions
    df = df[df['Description'].notna() & df['Description'].str.strip().astype(bool)]
    
    # Remove duplicate descriptions
    df = df.drop_duplicates(subset=['Description'])
        
    # Remove records where the description contains the word "rejected"
    df = df[~df['Description'].str.contains('rejected', case=False, na=False)]
    
    return df

def main():
    # Path to the input CSV file
    input_csv_path = "/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/cve_data.csv"
    
    # Path to the output CSV file
    output_csv_path = "/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/cleaned_cve_data.csv"
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)
    
    # Clean the data
    cleaned_df = clean_cve_data(df)
    
    # Save the cleaned DataFrame to a new CSV file
    cleaned_df.to_csv(output_csv_path, index=False)
    
    print(f"Cleaned data saved to {output_csv_path}")

if __name__ == "__main__":
    main()
