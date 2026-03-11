#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:27:25 2024

@author: mariafedericanorelli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:53:15 2024

Python script to retrieve data from a specified URL or locally stored, 
read the data into pandas DataFrames, creates a network graph using NetworkX, 
and then saves the graph data as a GraphML, GML or GEXF file.

@author: molly
"""

from collections import defaultdict
import shutil
import numpy as np
import sys
import requests
import pandas as pd
import networkx as nx
import urllib.request
import os
import csv 
import datetime 
import matplotlib.pyplot as plt


_DATACACHE = "~/polygraphs-cache/data"

def validate_origin(origin):
    """
    Validates the origin, ensuring it's either a URL or a local file.

    Args:
        origin (str): The origin URL or local file path.

    Returns:
        bool: True if the origin is a URL, False if it's a local file.

    Raises:
        ValueError: If the origin is not a string.
        Exception: If the origin is neither a URL nor a local file, 
                   or if a URL is unreachable, or if the local file does not exist.
    """
    # Check if origin is a string
    if not isinstance(origin, str):
        raise ValueError("File origin must be a string")

    # Check if origin is a URL
    remote = urllib.parse.urlparse(origin).scheme is not None

    if not remote:  # If it's not a URL, treat it as a local file path
        if not os.path.isfile(origin):  # Check if the file exists
            raise Exception(f"Local file does not exist: {origin}")

    return remote  # Return True if it's a URL, False if it's a local file

def fetch_file(origin, destination_folder):
    """
    Downloads remote origin to destination folder or copies local file to folder.
    """
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist

    if urllib.parse.urlparse(origin).scheme:
        # If remote file, download to destination folder
        filename = os.path.basename(urllib.parse.urlparse(origin).path)
        destination_path = os.path.join(destination_folder, filename)
        response = requests.get(origin)
        with open(destination_path, 'wb') as f:
            f.write(response.content)
    else:
        # If local file, copy to destination folder
        # Remove 'file://' scheme if present
        origin = urllib.parse.urlparse(origin).path if origin.startswith('file://') else origin
        filename = os.path.basename(origin)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copyfile(origin, destination_path)
    return destination_path

def txt_to_csv(input_file, output_file):
    """
    Converts a text file with ';' delimiter to a CSV file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
        txt_reader = csv.reader(txt_file, delimiter=';')
        csv_writer = csv.writer(txt_reader)

        for row in txt_reader:
            csv_writer.writerow(row)

    print(f"Conversion completed. CSV file saved at: {output_file}")

def read_csv(file, source_column_name, destination_column_name):
    """Reads csv and inspects it to get source and destination for edges"""
    df = pd.read_csv(file)  # Read the CSV file
    
    # Check if values in the first column are space-separated
    try: 
        if df.iloc[:, 0].str.contains(' ').any():
        # If space-separated, split the values into two columns
            split_values = df.iloc[:, 0].str.split(' ', n=1, expand=True)
            df[source_column_name] = split_values[0]
            df[destination_column_name] = split_values[1]
            df = df.drop(columns=[df.columns[0]])  # Drop the original first column

    except AttributeError:
        pass  # Do nothing if AttributeError occurs
    # Assign source and destination columns
    source = df[source_column_name]
    destination = df[destination_column_name]

    return df, source, destination

def normalize_graph(graph):
    """Normalize all numerical attributes in the graph and keep original values as labels."""
    # Create a dict with original id
    original_id = dict(zip(list(graph.nodes()), list(graph.nodes())))
    # Set the original_id dictionary as node attributes
    nx.set_node_attributes(graph, original_id, "original_id")

    # Normalize node identifiers (from 0 to N) using default dict
    tbl = defaultdict(lambda: len(tbl))
    normalized_node_edges = [(tbl[edge[0]], tbl[edge[1]]) for edge in graph.edges()]

    # Relabel nodes using lookup table
    normalized_graph = nx.relabel_nodes(graph, tbl)

    # Extract weights from the original graph
    original_weights = [data['weight'] for _, _, data in graph.edges(data=True) if 'weight' in data]

    if original_weights:
        min_weight = min(original_weights)
        max_weight = max(original_weights)
        
        for u, v, data in normalized_graph.edges(data=True):
            if 'weight' in data:
                original_weight = data['weight']
                normalized_weight = (original_weight - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 0
                data['weight'] = normalized_weight
                data['original_weight'] = original_weight

    return normalized_graph

def create_graph(source, destination):
    """Create a graph based on the source and destination columns."""
    G = nx.DiGraph()
    for src, dst_str in zip(source, destination):
        # Check if the destination is a string
        if isinstance(dst_str, str):
            # Split the destination node string into node ID and weight
            node_data = dst_str.split()
            dst_node_id = int(node_data[0])  # Extract the destination node ID
            # Check if weight is available
            if len(node_data) > 1:
                weight = int(node_data[1])  # Extract the weight
                G.add_edge(int(src), dst_node_id, weight=weight)  # Add edge with weight as attribute
            else:
                G.add_edge(int(src), dst_node_id)  # Add edge without weight
        # If the destination is a tuple of integers (node value, weight)
        elif isinstance(dst_str, tuple) and len(dst_str) == 2:
            dst_node_id, weight = dst_str
            G.add_edge(int(src), dst_node_id, weight=weight)  # Add edge with weight as attribute
        # If the destination is an integer
        elif isinstance(dst_str, int):
            G.add_edge(int(src), dst_str)
        else:
            print(f"Warning: Unexpected destination type for node '{src}': {type(dst_str)}")

    print("The graph has", len(G), "nodes and", len(G.edges()), "edges")
    return G

def save_graph(graph, format_choice, destination_folder, filename):
    """
    Saves the graph in the specified format (GML, GraphML, or GEXF) to the destination folder.

    """
    filepath = os.path.join(destination_folder, filename + "." + format_choice.lower())

    if format_choice.lower() == "gml":
        nx.write_gml(graph, filepath)
        print(f"Graph saved as {filepath}")
    elif format_choice.lower() == "graphml":
        nx.write_graphml(graph, filepath)
        print(f"Graph saved as {filepath}")
    elif format_choice.lower() == "gexf":
        nx.write_gexf(graph, filepath)
        print(f"Graph saved as {filepath}")
    else:
        print("Invalid file format. Please choose either GML, GraphML, or GEXF.")

def print_graph_details(graph, num_nodes=20, num_edges=20, output_file="subgraph.png"):
    """
    Print details of a subset of nodes and edges and save the subgraph as an image.
    """
    print("\nGraph Details:")
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")

    print("\nSample nodes:")
    subset_nodes = []
    for i, node in enumerate(graph.nodes(data=True)):
        if i >= num_nodes:
            break
        print(node)
        subset_nodes.append(node[0])  # Add node to subset

    print("\nSample edges:")
    subset_edges = []
    for i, edge in enumerate(graph.edges(data=True)):
        if i >= num_edges:
            break
        print(edge)
        subset_edges.append((edge[0], edge[1]))  # Add edge to subset

    # Create a subgraph
    subgraph = graph.edge_subgraph(subset_edges).copy()

    # Draw the subgraph using spring layout
    plt.figure(figsize=(12, 12))  # Set larger figure size
    pos = nx.spring_layout(subgraph)  # Use spring layout
    nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=15, font_weight='bold')

    # Save the subgraph as an image
    plt.savefig(output_file)
    print(f"\nSubgraph saved as {output_file}")

def extract_dataset_name(filepath):
    """
    Extracts the dataset name from the file path.
    """
    return os.path.splitext(os.path.basename(filepath))[0]


#Examples (use base, source and dest as inputs)
#Ex1 Bacon online dataset 
#Get today's date
# dt = datetime.datetime.now()
# date = "_{y}_{m:02d}_{d:02d}.csv".format(y=dt.year, m=dt.month, d=dt.day)
# base_undated = 'http://sixdegreesoffrancisbacon.com/data/SDFB_relationships'
# base = base_undated + date # input data
# source = 'person1_index' # input source
# dest = 'person2_index' #input dest

# wikipedia dataset locally saved 
base ='file:///Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/tkgl-smallpedia/tkgl-smallpedia_base.csv'
source= 'head'  #  column name for source
dest = 'tail'  # column name for destination



if __name__ == "__main__":
    # Dictionary to map variable names to their values
    predefined_vars = {
        'base': base,
        'source': source,
        'dest': dest
    }

    # User input for variable names
    origin = input("Enter URL, local file path, or private repository URL (or predefined variable name): ")
    source_column_name = input("Enter the name of the source column (or predefined variable name): ")
    destination_column_name = input("Enter the name of the destination column (or predefined variable name): ")

    # Retrieve the values of the variables if they are predefined variable names
    origin = predefined_vars.get(origin, origin)
    source_column_name = predefined_vars.get(source_column_name, source_column_name)
    destination_column_name = predefined_vars.get(destination_column_name, destination_column_name)

    destination_folder = os.path.expanduser(_DATACACHE)

    # Fetch the file based on the origin
    downloaded_file = None  # Initialize downloaded_file to None
    if origin.startswith("file://"):  # Check if it's a local file path
        downloaded_file = urllib.parse.urlparse(origin).path  # Extract the file path
    # Check if the downloaded file is already a CSV, otherwise convert it to CSV
        if downloaded_file.endswith('.csv'):
            print("Input file is already a CSV file.")
        else:
            output_file = downloaded_file.replace('.txt', '.csv')  # Define output_file here
            if not os.path.exists(output_file):
                txt_to_csv(downloaded_file, output_file)
                downloaded_file = output_file
            else:
                print(f"Output file {output_file} already exists. Please choose a different output file name.")
    elif os.path.isfile(origin):  # Check if it's a local file path without the "file://" scheme
        downloaded_file = origin
    # Check if the downloaded file is already a CSV, otherwise convert it to CSV
        if downloaded_file.endswith('.csv'):
            print("Input file is already a CSV file.")
        else:
            output_file = downloaded_file.replace('.txt', '.csv')  # Define output_file here
        if not os.path.exists(output_file):
            txt_to_csv(downloaded_file, output_file)
            downloaded_file = output_file
        else:
            print(f"Output file {output_file} already exists. Please choose a different output file name.")

    elif os.path.isfile(origin):  # Check if it's a local file path without the "file://" scheme
        downloaded_file = origin
        # Check if the downloaded file is already a CSV, otherwise convert it to CSV
        if downloaded_file.endswith('.csv'):
            print("Input file is already a CSV file.")
        else:
            output_file = downloaded_file.replace('.txt', '.csv')
            if not os.path.exists(output_file):
                txt_to_csv(downloaded_file, output_file)
                downloaded_file = output_file
            else:
                print("Output file already exists. Please choose a different output file name.")
    else:
    # If it's not a local file path, treat it as a URL and proceed with the existing code
        is_remote = validate_origin(origin)
        if is_remote:
            try:
                downloaded_file = fetch_file(origin, destination_folder)
                print("Downloaded file:", downloaded_file)

            # Check if the downloaded file is already a CSV, otherwise convert it to CSV
                if downloaded_file.endswith('.csv'):
                    print("Input file is already a CSV file.")
                else:
                    output_file = downloaded_file.replace('.txt', '.csv')
                    if not os.path.exists(output_file):
                        txt_to_csv(downloaded_file, output_file)
                        downloaded_file = output_file
                    else:
                        print("Output file already exists. Please choose a different output file name.")

            except Exception as e:
                print("Error:", e)
                sys.exit(1)  # Exit the script if there's an error downloading the file
        else:
            downloaded_file = origin

    # Extract dataset name
    dataset_name = extract_dataset_name(downloaded_file)
    
    # Read the CSV file and assign source and destination columns
    try:
        df, source, destination = read_csv(downloaded_file, source_column_name, destination_column_name)   
        if df is not None:
            print("Source column:", source_column_name)
            print("Destination column:", destination_column_name)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)  # Exit the script if there's an error reading the CSV file
    
    # Create the graph
    try:
        graph = create_graph(source, destination)
        print("Graph created successfully.")
        print_graph_details(graph, num_nodes=50, num_edges=50, output_file=f"{dataset_name}_subgraph_non_normalized.png")  # Print details of the graph

        # Save the non-normalized graph
        save_graph(graph, "GML", destination_folder, f"{dataset_name}_graph_non_normalized")

        # Normalize the graph
        normalized_graph = normalize_graph(graph)
        print("Graph normalized successfully.")
        
        # Print and save the normalized graph
        print_graph_details(normalized_graph, num_nodes=50, num_edges=50, output_file=f"{dataset_name}_subgraph_normalized.png")  # Print details of the graph
        save_graph(normalized_graph, "GML", destination_folder, f"{dataset_name}_graph_normalized")

    except Exception as e:
        print("Error:", e)
        sys.exit(1)  # Exit the script if there's an error creating or saving the graph
        

        
