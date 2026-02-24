import json
import csv
from io import StringIO
import os

stats = []


def set_stats(val1, val2="", val3="", val4="", val5="", val6="", val7="", val8=""):
    global stats
    
    # Start with val1 (always present since it's required)
    record = [val1]
    
    # Add each subsequent value only if it's present (non-empty)
    for val in [val2, val3, val4, val5, val6, val7, val8]:
        if val:  # Only add non-empty values
            record.append(val)
        else:
            break  # Stop at first empty value to maintain order
    
    # Append the record to stats
    stats.append(record)
    return stats

def get_stats():
    return stats


def stats_json_to_csv_text(table_data):
    output = StringIO()
    writer = csv.writer(output)
    
    # Process your table structure here
    for row in table_data:
        # Transform row based on your specific structure
        writer.writerow(row)
    
    return output.getvalue()

def stats_save_json_to_csv(json_data_text, output_file):    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write(json_data_text)
    print(f"CSV saved to {output_file}")


def clean_file_names( file_names, txt: str):
    new_txt = txt
    for fn in file_names:
        fnb = os.path.splitext(fn)[0]
        new_txt = new_txt.replace( fnb, "" )
    return new_txt