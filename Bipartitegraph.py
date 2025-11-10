import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data():
    """
    Load the Santa Barbara businesses, reviews, and users data
    """
    reviews_df = pd.read_csv('santa_barbara_reviews.csv')
    users_df = pd.read_csv('santa_barbara_users.csv')
    return reviews_df, users_df

def create_user_business_mapping(reviews_df):
    """
    Create a mapping of businesses to their reviewers
    """
    business_to_users = defaultdict(set)
    for _, review in reviews_df.iterrows():
        business_to_users[review['business_id']].add(review['user_id'])
    return business_to_users

def create_user_network(business_to_users):
    """
    Create a network where users are connected if they reviewed the same business
    """
    G = nx.Graph()
    
    # Add edges between users who reviewed the same business
    for business_id, users in business_to_users.items():
        # Convert users set to list for indexing
        users_list = list(users)
        # Add edges between all pairs of users who reviewed this business
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                user1, user2 = users_list[i], users_list[j]
                # If edge already exists, increment weight
                if G.has_edge(user1, user2):
                    G[user1][user2]['weight'] += 1
                else:
                    # Create new edge with weight 1
                    G.add_edge(user1, user2, weight=1)
    
    return G

def analyze_network(G):
    """
    Analyze the network and print basic statistics
    """
    print("\nNetwork Statistics:")
    print(f"Number of users (nodes): {G.number_of_nodes():,}")
    print(f"Number of connections (edges): {G.number_of_edges():,}")
    
    if G.number_of_nodes() > 0:
        # Calculate degree statistics
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees)
        print(f"Average number of connections per user: {avg_degree:.2f}")
        
        # Find users with most connections
        degree_dict = dict(G.degree())
        top_users = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 users by number of connections:")
        for user_id, degree in top_users:
            print(f"User {user_id}: {degree} connections")
        
        # Calculate network density
        density = nx.density(G)
        print(f"\nNetwork density: {density:.4f}")
        
        # Find number of connected components
        num_components = nx.number_connected_components(G)
        print(f"Number of connected components: {num_components}")
        
        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Size of largest connected component: {len(largest_cc):,} users")

def visualize_network(G, output_file='user_network.png'):
    """
    Create and save a visualization of the network
    Only visualize a subset of the network if it's too large
    """
    if G.number_of_nodes() > 100:
        # If network is too large, visualize only the most connected subset
        degrees = dict(G.degree())
        top_users = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
        top_user_ids = {user_id for user_id, _ in top_users}
        H = G.subgraph(top_user_ids)
    else:
        H = G

    plt.figure(figsize=(12, 8))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(H)
    
    # Draw the network
    edges = H.edges()
    weights = [H[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_nodes(H, pos, node_size=100, node_color='lightblue')
    nx.draw_networkx_edges(H, pos, width=[w/max(weights) for w in weights], 
                          alpha=0.5, edge_color='gray')
    
    plt.title("User Network in Santa Barbara\n(Users connected by reviewing same businesses)")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nNetwork visualization saved as '{output_file}'")

def main():
    # Load the data
    print("Loading data...")
    reviews_df, users_df = load_data()
    
    # Create business to users mapping
    print("Creating business to users mapping...")
    business_to_users = create_user_business_mapping(reviews_df)
    
    # Create the network
    print("Creating user network...")
    G = create_user_network(business_to_users)
    
    # Analyze the network
    analyze_network(G)
    
    # Visualize the network
    print("\nCreating network visualization...")
    visualize_network(G)
    
    # Save the network data
    print("\nSaving network data...")
    # Save edge list with weights
    nx.write_weighted_edgelist(G, "user_network.edgelist")
    print("Network data saved as 'user_network.edgelist'")

if __name__ == "__main__":
    main()
