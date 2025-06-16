import subprocess
import sys
import time
import csv
import os

# The question to use for all benchmarks (edit as needed or pass via sys.argv)
question = "I'd like to compare hyperscalers to assess which one is the best choice for enterprise use, in about 600 words?"

# Output CSV filenames for each script
azure_csv = "azure_llama_results.csv"
gcp_csv = "gcp_llama_results.csv"
aws_csv = "aws_llama_results.csv"

# Paths to your scripts
azure_script = "azure_llama_demo.py"
gcp_script = "gcp_llama_demo.py"
aws_script = "aws_llama_demo.py"

# List of scripts to run with their CSV filenames and provider names
scripts = [
    ("Azure Llama", azure_script, azure_csv),
    ("GCP Llama", gcp_script, gcp_csv),
    ("AWS Bedrock Llama", aws_script, aws_csv),
]

start_time = time.time()  # Start timing

for name, script, csv_file in scripts:
    print(f"\n=== Running {name} Benchmark ===\n")
    try:
        subprocess.run(
            [sys.executable, script, "--question", question, "--csv", csv_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

end_time = time.time()  # End timing
elapsed = end_time - start_time

print(f"\nAll benchmarks completed in {elapsed:.2f} seconds.")

# Compile averages from each CSV into a summary file
summary_csv = "benchmark_summary.csv"
summary_rows = []
header = [
    "Provider",
    "Average Response Time (s)",
    "Average Prompt Tokens",
    "Average Completion Tokens",
    "Average Total Tokens",
    "Average Characters",
    "Average Words",
    "Average Cost",
    "Region"
]

for name, _, csv_file in scripts:
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping.")
        continue
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Find the row that starts with "Average"
        avg_row = next((row for row in rows if row and row[0].strip().lower() == "average"), None)
        # Find the region column index from the header
        region_index = None
        for row in rows:
            if row and "region" in [col.strip().lower() for col in row]:
                region_index = [col.strip().lower() for col in row].index("region")
                break
        # Get the region value from the averages row if present, else from the first data row
        region_value = ""
        if avg_row and region_index is not None and len(avg_row) > region_index:
            region_value = avg_row[region_index]
        else:
            # Try to get from the first data row if not in averages
            data_row = next((row for row in rows if row and row[0].strip().isdigit()), None)
            if data_row and region_index is not None and len(data_row) > region_index:
                region_value = data_row[region_index]
        if avg_row:
            # Only keep the relevant columns (including cost, which should be at index 7)
            summary_rows.append([name] + avg_row[1:8+1] + [region_value])
        else:
            print(f"Warning: No averages found in {csv_file}")

# Write the summary CSV
with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in summary_rows:
        writer.writerow(row)

print(f"\nSummary written to {summary_csv}")

# Transpose the summary so metrics are rows and providers are columns
provider_names = [row[0] for row in summary_rows]
averages_by_provider = [row[1:] for row in summary_rows]

metric_names = [
    "Avg. Response Time (s)",
    "Avg. Prompt Tokens",
    "Avg. Completion Tokens",
    "Avg. Total Tokens",
    "Avg. Characters",
    "Avg. Words",
    "Avg. Cost",
    "Region"
]

transposed_rows = []
header_row = ["Metric"] + provider_names
transposed_rows.append(header_row)

for i, metric in enumerate(metric_names):
    row = [metric]
    for provider_avg in averages_by_provider:
        value = provider_avg[i] if i < len(provider_avg) else ""
        row.append(value)
    transposed_rows.append(row)

with open("benchmark_summary_transposed.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in transposed_rows:
        writer.writerow(row)

print("\nTransposed summary written to benchmark_summary_transposed.csv")