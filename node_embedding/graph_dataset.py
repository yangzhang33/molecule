from tqdm import tqdm
import os

import torch

from torch_geometric.data import InMemoryDataset, Data

def read_graph_file_and_convert_to_tensors(file_path):
    # Initialize lists to hold edge list, edge attributes, and node attributes
    edge_list = []
    edge_attributes = []
    node_attributes = []
    
    # Initialize a variable to track the current section of the file
    current_section = None
    
    # Open and read the .graph file
    with open(file_path, 'r') as file:
        for line in file:
            # Check the current section
            if line.startswith('# Edge List'):
                current_section = 'edge_list'
                continue
            elif line.startswith('# Edge Attributes'):
                current_section = 'edge_attributes'
                continue
            elif line.startswith('# Node Attributes'):
                current_section = 'node_attributes'
                continue
            
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            # Based on the current section, process and append the data accordingly
            if current_section == 'edge_list':
                edge_list.append(list(map(int, line.strip().split())))
            elif current_section == 'edge_attributes':
                edge_attributes.append(list(map(int, line.strip().split())))
            elif current_section == 'node_attributes':
                node_attributes.append(list(map(int, line.strip().split())))
    
    # Convert lists to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    x = torch.tensor(node_attributes, dtype=torch.float)
    y = torch.zeros((1, 0), dtype=torch.float)
    
    return x, y, edge_attr, edge_index



class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Dynamically list all .graph files in the raw_dir
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.graph')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download your `.graph` files here if necessary.
        # Since you mentioned having a repo, you might clone it or copy files here.
        pass

    def process(self):
        data_list = []
        for raw_path in tqdm(self.raw_paths, desc="Processing"):
            # Your function to load a .graph file and return x, y, edge_attr, edge_index
            x, y, edge_attr, edge_index = read_graph_file_and_convert_to_tensors(os.path.join(self.raw_dir, raw_path))
            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])