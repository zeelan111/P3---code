import networkx as nx
import random

def create_smaller_sample(node_amount:int, path:str, save = False, paper_name = 'paper2paper_2000_gcc'): # author_name = 'author_sample'
    
    paper: nx.Graph = nx.read_gml(path + paper_name + '.gml')
    print('paper loaded with', paper.number_of_nodes(), 'nodes and', paper.number_of_edges(), 'edges.', "it is a", type(paper))

    # remove edges to same node
    for node in paper.nodes():
        neighbors = list(paper.neighbors(node))
        if node in neighbors:
            paper.remove_edge(node, node)
    
    itteration = 1000
    while paper.number_of_nodes() > node_amount and itteration > 0:
        node = random.choice(list(paper.nodes))
        neighbors = list(paper.successors(node))  # Outgoing neighbors
        predecessors = list(paper.predecessors(node))  # Incoming neighbors
        connected = nx.is_weakly_connected(paper)
        
        if not connected:
            itteration -= 1
            paper.add_node(node)
            # Re-add outgoing edges
            for neighbor in neighbors:
                paper.add_edge(node, neighbor)

            # Re-add incoming edges
            for predecessor in predecessors:
                paper.add_edge(predecessor, node)
        
        elif connected:
            print(f'removed node: {node}')
            itteration = 1000



    if save == True:
        nx.write_gml(paper, path + 'paper_new_sample_made_in_ailab_69.gml')
        print("Saved")


# Insert your path here!!!!!!!!!
path = path = r''
create_smaller_sample(100000, path, save=True)