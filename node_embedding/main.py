from graph_dataset import MyGraphDataset
from graph_transformer import GraphTransformer

import torch.nn as nn

import utils
from torch_geometric.data import DataLoader


def main():
    # same as Digress default
    n_layers = 5
    hidden_mlp_dims={'X': 256, 'E': 128, 'y': 128}
    hidden_dims={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}

    # example
    dataset = MyGraphDataset(root = '/root/MolSTM_data_version2.1')
    print(f"Total number of graphs: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    example_batch = next(iter(data_loader))
    print(example_batch)
    # print(example_batch.edge_index)

    # same input output dims.
    input_dims, output_dims = utils.calculate_input_output_dimensions(dataset)

    model = GraphTransformer(n_layers=n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=hidden_mlp_dims,
                                    hidden_dims=hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU())
    
    dense_data, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr, example_batch.batch)
    dense_data = dense_data.mask(node_mask)
    X, E = dense_data.X, dense_data.E

    print(X.shape, E.shape, example_batch.y.shape)
    aggregated_feature, X, E, y = model(X, E, example_batch.y, node_mask)

    print(aggregated_feature[0])
if __name__ == '__main__':
    main()