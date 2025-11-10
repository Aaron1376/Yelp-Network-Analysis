import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data():
    """
    Load the Santa Barbara restaurants, restaurant reviews, and users data
    """
    restaurants_df = pd.read_csv('santa_barbara_restaurants.csv')
    reviews_df = pd.read_csv('santa_barbara_restaurant_reviews.csv')
    users_df = pd.read_csv('santa_barbara_users.csv')
    return restaurants_df, reviews_df, users_df

def create_user_business_mapping(reviews_df):
    """
    Create a mapping of restaurants to their reviewers
    """
    restaurant_to_users = defaultdict(set)
    for _, review in reviews_df.iterrows():
        restaurant_to_users[review['business_id']].add(review['user_id'])
    return restaurant_to_users

def create_user_network(restaurant_to_users):
    """
    Create a network where users are connected if they reviewed the same restaurant
    """
    G = nx.Graph()
    
    # Add edges between users who reviewed the same restaurant
    for restaurant_id, users in restaurant_to_users.items():
        # Convert users set to list for indexing
        users_list = list(users)
        # Add edges between all pairs of users who reviewed this restaurant
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

def visualize_network(G, output_file='restaurant_user_network.png'):
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
    
    plt.title("User Network in Santa Barbara Restaurants\n(Users connected by reviewing same restaurants)")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nNetwork visualization saved as '{output_file}'")

def main():
    # Load the data
    print("Loading restaurant data...")
    restaurants_df, reviews_df, users_df = load_data()
    
    print(f"Loaded {len(restaurants_df):,} restaurants")
    print(f"Loaded {len(reviews_df):,} restaurant reviews")
    print(f"Loaded {len(users_df):,} users")
    
    # Create restaurant to users mapping
    print("Creating restaurant to users mapping...")
    restaurant_to_users = create_user_business_mapping(reviews_df)
    
    # Create the network
    print("Creating user network...")
    G = create_user_network(restaurant_to_users)
    
    # Analyze the network
    analyze_network(G)
    
    # Visualize the network
    print("\nCreating network visualization...")
    visualize_network(G)
    
    # Save the network data
    print("\nSaving network data...")
    # Save edge list with weights
    nx.write_weighted_edgelist(G, "restaurant_user_network.edgelist")
    print("Network data saved as 'restaurant_user_network.edgelist'")
    
    # Additional restaurant-specific analysis
    print("\n" + "="*60)
    print("RESTAURANT-SPECIFIC ANALYSIS")
    print("="*60)
    
    # Find restaurants with most shared reviewers
    shared_reviewers = {}
    restaurant_names = dict(zip(restaurants_df['business_id'], restaurants_df['name']))
    
    for restaurant_id, users in restaurant_to_users.items():
        if len(users) > 1:  # Only consider restaurants with multiple reviewers
            shared_reviewers[restaurant_id] = len(users)
    
    top_shared = sorted(shared_reviewers.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 restaurants by number of reviewers:")
    for i, (restaurant_id, user_count) in enumerate(top_shared, 1):
        name = restaurant_names.get(restaurant_id, "Unknown")
        print(f"{i:2d}. {name:<35} {user_count:>4} reviewers")
    
    return G, restaurants_df, reviews_df, users_df

if __name__ == "__main__":
    main()
