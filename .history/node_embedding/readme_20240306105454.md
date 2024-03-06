# How to use:
* download the data MolSTM_data_version2.1 and unzip it all.
* Change the root for dataset in main.py
* Run python main.py.

main.py runs the model for one batch, its inputs are: 
* x: the node attributes
* edge_index: the edge list
* edge_attr: the edge attributes
* y: the global feature, empty here.

the outputs are: 
* aggregated_features: the vector embedding of node features, the aggregation is sum over x.
* x: the updated node attributes
* edge_index: the updated edge list
* edge_attr: the updated edge attributes
* y: the updated global feature, empty here.

the input and output dimensions are the same, if we introduce extra node features, it could be different.