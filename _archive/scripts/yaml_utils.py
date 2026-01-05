#!/usr/bin/env python3
"""
YAML Processor for extracting task information and generating key mappings.

This script reads a YAML metadata file, finds a specific task by name,
extracts primary key columns, and generates a new YAML with key mappings
and task metadata.
"""

import yaml
import argparse
import sys
from pathlib import Path


def load_yaml(file_path):
    """Load YAML file and return the parsed data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def find_task_by_name(data, task_name):
    """Find a task in the YAML data by its name."""
    if 'tasks' not in data:
        print("Error: No 'tasks' key found in the YAML file.")
        return None
    
    for task in data['tasks']:
        if task.get('name') == task_name:
            return task
    
    print(f"Error: Task '{task_name}' not found in the YAML file.")
    return None

def get_task_names(data):
    """Get all task names from the YAML data."""
    task_names = []
    if 'tasks' not in data:
        print("Error: No 'tasks' key found in the YAML file.")
        return None
    
    for task in data['tasks']:
        task_names.append(task.get('name'))
    return task_names

def extract_primary_keys(columns):
    """Extract all columns with dtype 'primary_key' and return their names."""
    primary_keys = []
    for column in columns:
        if column.get('dtype') == 'primary_key':
            primary_keys.append(column.get('name'))
    return primary_keys


def generate_key_mappings(primary_keys, target_table):
    """Generate key mappings in the format {name}: {target_table}.{name}"""
    key_mappings = {}
    for key_name in primary_keys:
        key_mappings[key_name] = f"{target_table}.{key_name}"
    return key_mappings


def process_yaml(input_path, task_names: list[str]=None):
    """Process the YAML file and generate the output structure."""
    # Load the input YAML
    data = load_yaml(input_path)

    output_data = []

    if task_names is None:
        task_names = get_task_names(data)

    for task_name in task_names:
        task = find_task_by_name(data, task_name)
        if task is None:
            print(f"Error: Task '{task_name}' not found in the YAML file.")
            continue
        
        # Extract required information
        columns = task.get('columns', [])
        target_table = task.get('target_table')
        target_column = task.get('target_column')
        time_column = task.get('time_column')
        task_type = task.get('task_type')
        
        # Extract primary keys
        primary_keys = extract_primary_keys(columns)
        
        # Generate key mappings
        key_mappings = generate_key_mappings(primary_keys, target_table)

        output_data.append({
            'key_mappings': key_mappings,
            'target_column': target_column,
            'time_column': time_column,
            'task_type': task_type,
            "task_name": task_name
        })

    return output_data


def main():
    """Main function to handle command line arguments and process the YAML."""
    parser = argparse.ArgumentParser(
        description='Process YAML metadata file and extract task information'
    )
    parser.add_argument(
        'input_path',
        help='Path to the input YAML metadata file'
    )
    parser.add_argument(
        'task_name',
        help='Name of the task to extract (e.g., "study-outcome")'
    )
    parser.add_argument(
        '--output',
        help='Path to the output YAML file (optional)'
    )
    
    args = parser.parse_args()
    
    # Process the YAML
    output_data = process_yaml(args.input_path, args.task_name)
    
    if output_data is None:
        sys.exit(1)
    
    # Convert to YAML string
    output_yaml = yaml.dump(output_data, default_flow_style=False, sort_keys=False)
    
    # Print or save the output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as file:
                file.write(output_yaml)
            print(f"Output saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            sys.exit(1)
    else:
        print("Generated YAML:")
        print(output_yaml)


if __name__ == "__main__":
    main()
