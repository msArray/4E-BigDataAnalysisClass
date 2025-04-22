import csv
import glob
import sys
import os
import re

def convert_text_to_csv(input_file, output_file, delimiter=','):
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        
        # Read the text file
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        if not lines:
            print("Warning: Input file is empty")
            return
        
        # Process each line and split by one or more spaces
        data = [re.split(r'\s+', line.strip()) for line in lines if line.strip()]
        
        # Check for consistent field count
        field_counts = [len(row) for row in data]
        if len(set(field_counts)) > 1:
            print("Warning: Input file has inconsistent number of fields per line")
        
        # Write to CSV
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=delimiter)
            writer.writerows(data)
            
        print(f"Successfully converted {len(data)} lines from '{input_file}' to '{output_file}'")
            
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    files = glob.glob("./scraped/*")
    for file in files:
        filename = re.search(r'[^\\\/]+$', file).group(0)
        print(filename)
        convert_text_to_csv(file, f'./out/{filename}', delimiter=',')
        print(f"Converted {file} to CSV format.")

if __name__ == "__main__":
    main()