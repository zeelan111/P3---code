import networkx as nx
import matplotlib.pyplot as plt
import random

path = r'C:/Users/hjort/OneDrive - Aalborg Universitet/AAU/3 semester/Project/data/'
#paper_sample.gml

class GRAPH_DATA:

    def __init__(self, path, name_paper): #name_author
        """Opens the data"""
        self.path = path
        self.paper: nx.Graph = nx.read_gml(self.path + name_paper +'.gml')
        #self.paper = nx.petersen_graph()
        print('paper loaded with', self.paper.number_of_nodes(), 'nodes and', self.paper.number_of_edges(), 'edges.', "it is a", type(self.paper))
        #self.author = nx.read_gml(self.path + name_author +'.gml')
        #print('author loaded with', self.author.number_of_nodes(), 'nodes and', self.author.number_of_edges(), 'edges.', "it is a", type(self.author))
        
        

    def create_smaller_sample(self, node_amount:int, save = False, paper_name = 'paper_sample'): # author_name = 'author_sample'
        print('creating a smaller dataset')
        
        # remove edges to same node
        for node in self.paper.nodes():
            neighbors = list(self.paper.neighbors(node))
            if node in neighbors:
                self.paper.remove_edge(node, node)
        
        itteration = 1000
        while self.paper.number_of_nodes() > node_amount and itteration > 0:
            node = random.choice(list(self.paper.nodes))
            #paper_temp = self.paper.copy()
            #paper_temp.remove_node(node)
            neighbors = list(self.paper.successors(node))  # Outgoing neighbors
            predecessors = list(self.paper.predecessors(node))  # Incoming neighbors
            connected = nx.is_weakly_connected(self.paper)
            
            if not connected:
                itteration -= 1
                self.paper.add_node(node)
                # Re-add outgoing edges
                for neighbor in neighbors:
                    self.paper.add_edge(node, neighbor)

                # Re-add incoming edges
                for predecessor in predecessors:
                    self.paper.add_edge(predecessor, node)
            
            elif connected:
                itteration = 1000


        """# Sort nodes by degree in ascending order
        nodes_by_degree = sorted(self.paper.degree, key=lambda x: x[1])
            
        # Nodes to remove
        nodes_to_remove = [node for node, degree in nodes_by_degree[:self.paper.number_of_nodes() - node_amount]]
            
        # Remove those nodes
        self.paper.remove_nodes_from(nodes_to_remove)"""

            
        



        # remove any nodes with 0 edges
        #for node in self.paper.nodes():
         #   if self.paper.degree(node) == 0:
          #      nodes_to_remove.append(node)
            
        
        #if len(nodes_to_remove) > 0:
         #   self.paper.remove_nodes_from(nodes_to_remove)


        print('Paper graph now with', self.paper.number_of_nodes(), 'nodes and', self.paper.number_of_edges(), 'edges.')

        

        #nodes_g1 = set(self.paper.nodes()) # all nodes in paper
        #nodes_to_remove = set(self.author.nodes()) # all nodes in author

        # Find the intersection of the two node sets
        #common_nodes = nodes_g1.intersection(nodes_to_remove)
        
        #neighbors_to_add = set() # holding nighbors of the nodes

        #for node in common_nodes:
           # in_neighbors = set(self.author.predecessors(node))
          #  out_neighbors = set(self.author.successors(node))
        
            # Combine both sets of neighbors
         #   neighbors = in_neighbors | out_neighbors
        #    neighbors_to_add.update(neighbors) # adds nighbors to be kept
        
        
       # common_nodes.update(neighbors_to_add) # adds all the nodes that share an edge with nodes alredy in the set
        
        # removes nodes that are to be kept from the list to be removed
        #for node in common_nodes:
         #   nodes_to_remove.discard(node)
        
        #self.author.remove_nodes_from(nodes_to_remove)

        # remove edges to same node
        """for node in self.author.nodes():
            neighbors = list(self.author.neighbors(node))
            if node in neighbors:
                self.author.remove_edge(node, node)"""
        
       # print('author graph now with', self.author.number_of_nodes(), 'nodes and', self.author.number_of_edges(), 'edges.')
        
        # saves data
        if save == True:
            nx.write_gml(self.paper, self.path + paper_name + '.gml')
            #nx.write_gml(self.author, self.path + author_name + '.gml')
            print("Saved")

    def show_graph(self): #graph
       """draw graph"""
       nx.draw(self.paper, with_labels=True, node_color='lightblue', edge_color='gray', node_size=100, font_size=5)
       plt.show()


G = GRAPH_DATA(path, 'paper2paper_2000_gcc',)
G.create_smaller_sample(100000 , save=True, paper_name="paper_new",)
#g = GRAPH_DATA(path, 'paper_draw')
#g.create_smaller_sample(100, save=False, paper_name="paper_draw")
#g.show_graph()

#neighbors = list(g.paper.neighbors("W2148143831"))

#print(f"Neighbors of w2148143831: {neighbors}")

