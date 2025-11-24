"""
Extended User Projection Network Analysis - 30 Users (Improved Connectivity)

This script creates a user projection network with 30 users that looks similar to the 15-user
network with better connectivity and visual density.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def create_extended_sample():
    """
    Create an extended sample with 30 users from the 500x500 sample, optimized for connectivity
    """
    print("Creating extended sample with 30 users (optimized for connectivity)...")
    
    # Read the full sample data (500x500)
    restaurants_df = pd.read_csv('santa_barbara_restaurants_sample.csv')
    reviews_df = pd.read_csv('santa_barbara_reviews_sample.csv')
    users_df = pd.read_csv('santa_barbara_users_sample.csv')
    
    print(f"Full sample: {len(restaurants_df)} restaurants, {len(users_df)} users, {len(reviews_df)} reviews")
    
    # First, find restaurants that have multiple reviews to ensure connectivity
    restaurant_review_counts = reviews_df.groupby('business_id').size().reset_index(name='review_count')
    popular_restaurants = restaurant_review_counts[restaurant_review_counts['review_count'] >= 2]
    popular_restaurant_ids = set(popular_restaurants['business_id'])
    
    # Get reviews for popular restaurants
    popular_reviews = reviews_df[reviews_df['business_id'].isin(popular_restaurant_ids)]
    
    # Select 30 users who have reviewed these popular restaurants (ensuring connectivity)
    user_review_counts = popular_reviews.groupby('user_id').size().reset_index(name='review_count')
    top_users = user_review_counts.nlargest(30, 'review_count')
    selected_user_ids = set(top_users['user_id'])
    
    # Filter users
    extended_users_df = users_df[users_df['user_id'].isin(selected_user_ids)]
    
    # Get all reviews for these users (not just popular restaurants)
    extended_reviews_df = reviews_df[reviews_df['user_id'].isin(selected_user_ids)]
    
    # Get restaurants that these users reviewed
    reviewed_restaurant_ids = set(extended_reviews_df['business_id'])
    extended_restaurants_df = restaurants_df[restaurants_df['business_id'].isin(reviewed_restaurant_ids)]
    
    # Save extended sample files
    extended_restaurants_df.to_csv('extended_restaurants_sample.csv', index=False)
    extended_users_df.to_csv('extended_users_sample.csv', index=False)
    extended_reviews_df.to_csv('extended_reviews_sample.csv', index=False)
    
    print(f"Extended sample created:")
    print(f"- Users: {len(extended_users_df)}")
    print(f"- Restaurants: {len(extended_restaurants_df)}")
    print(f"- Reviews: {len(extended_reviews_df)}")
    
    return extended_restaurants_df, extended_reviews_df, extended_users_df


def create_bipartite_graph(reviews_df, restaurants_df, users_df):
    """
    Create a bipartite graph connecting users to restaurants they reviewed
    """
    print("Creating bipartite graph...")
    B = nx.Graph()
    
    # Add restaurant nodes
    restaurant_names = dict(zip(restaurants_df['business_id'], restaurants_df['name']))
    for business_id in restaurants_df['business_id']:
        B.add_node(f"R_{business_id}", bipartite=0, type='restaurant', 
                  name=restaurant_names.get(business_id, "Unknown"))
    
    # Add user nodes
    user_names = dict(zip(users_df['user_id'], users_df.get('name', users_df['user_id'])))
    for user_id in users_df['user_id']:
        B.add_node(f"U_{user_id}", bipartite=1, type='user', 
                  name=user_names.get(user_id, f"User_{user_id[:8]}"))
    
    # Add edges based on reviews
    for _, review in reviews_df.iterrows():
        business_id = review['business_id']
        user_id = review['user_id']
        rating = review.get('stars', 0)
        
        B.add_edge(f"U_{user_id}", f"R_{business_id}", 
                  weight=1, rating=rating)
    
    print(f"Bipartite graph created: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    return B


def project_to_user_network(B):
    """
    Project the bipartite graph to create a user-user network
    """
    print("Projecting bipartite graph onto user set...")
    
    # Get user nodes
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    
    # Create projection onto user nodes
    user_graph = nx.bipartite.projected_graph(B, user_nodes)
    
    # Add additional attributes to edges
    for u1, u2 in user_graph.edges():
        # Find shared restaurants
        u1_restaurants = set(B.neighbors(u1))
        u2_restaurants = set(B.neighbors(u2))
        shared_restaurants = u1_restaurants & u2_restaurants
        
        # Calculate average ratings for shared restaurants
        shared_ratings_u1 = [B[u1][r]['rating'] for r in shared_restaurants if 'rating' in B[u1][r]]
        shared_ratings_u2 = [B[u2][r]['rating'] for r in shared_restaurants if 'rating' in B[u2][r]]
        
        avg_rating_u1 = np.mean(shared_ratings_u1) if shared_ratings_u1 else 0
        avg_rating_u2 = np.mean(shared_ratings_u2) if shared_ratings_u2 else 0
        avg_rating_diff = abs(avg_rating_u1 - avg_rating_u2)
        
        # Add edge attributes
        user_graph[u1][u2]['shared_restaurants'] = len(shared_restaurants)
        user_graph[u1][u2]['avg_rating_similarity'] = 5 - avg_rating_diff
        user_graph[u1][u2]['weight'] = len(shared_restaurants)
    
    print(f"User projection created: {user_graph.number_of_nodes()} users, {user_graph.number_of_edges()} connections")
    return user_graph


def analyze_extended_user_network(user_graph, B):
    """
    Analyze the extended user projection network
    """
    print("\n" + "="*60)
    print("EXTENDED USER PROJECTION NETWORK ANALYSIS (30 USERS)")
    print("="*60)
    
    print(f"Number of users: {user_graph.number_of_nodes():,}")
    print(f"Number of connections: {user_graph.number_of_edges():,}")
    
    if user_graph.number_of_nodes() > 0:
        # Calculate density
        density = nx.density(user_graph)
        print(f"Network density: {density:.3f}")
        
        # Degree statistics
        degrees = [d for n, d in user_graph.degree()]
        avg_degree = np.mean(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
        print(f"Average connections per user: {avg_degree:.2f}")
        print(f"Maximum connections: {max_degree}")
        print(f"Minimum connections: {min_degree}")
        
        # Find most connected users
        degree_dict = dict(user_graph.degree())
        top_users = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most connected users:")
        for user_node, degree in top_users:
            name = B.nodes[user_node]['name']
            print(f"  {name}: {degree} connections")
        
        # Network structure analysis
        if nx.is_connected(user_graph):
            diameter = nx.diameter(user_graph)
            avg_path_length = nx.average_shortest_path_length(user_graph)
            print(f"\nNetwork structure:")
            print(f"  Network diameter: {diameter}")
            print(f"  Average path length: {avg_path_length:.2f}")
        else:
            components = list(nx.connected_components(user_graph))
            print(f"\nNetwork has {len(components)} connected components")
            largest_component = max(components, key=len)
            print(f"  Largest component size: {len(largest_component)} users")
        
        # Clustering analysis
        clustering = nx.average_clustering(user_graph)
        print(f"  Average clustering coefficient: {clustering:.3f}")
        
        # Analyze shared restaurant patterns
        shared_counts = [user_graph[u1][u2]['shared_restaurants'] for u1, u2 in user_graph.edges()]
        if shared_counts:
            avg_shared = np.mean(shared_counts)
            max_shared = max(shared_counts)
            print(f"\nShared restaurant statistics:")
            print(f"  Average shared restaurants per connection: {avg_shared:.2f}")
            print(f"  Maximum shared restaurants: {max_shared}")
        
        # Rating similarity analysis
        similarities = [user_graph[u1][u2]['avg_rating_similarity'] for u1, u2 in user_graph.edges()]
        if similarities:
            avg_similarity = np.mean(similarities)
            print(f"  Average rating similarity: {avg_similarity:.2f}/5.0")


def visualize_extended_user_network(user_graph, B, output_file='extended_user_network_improved.png'):
    """
    Create a visualization similar to the 15-user network but with 30 users
    """
    print(f"\nCreating extended user network visualization...")
    
    plt.figure(figsize=(20, 16))
    
    # Create layout with better spacing - similar to the 15-user network
    pos = nx.spring_layout(user_graph, k=2.5, iterations=150, seed=42)
    
    # Prepare edge attributes
    edges = user_graph.edges(data=True)
    edge_weights = [d['shared_restaurants'] for u, v, d in edges]
    edge_similarities = [d['avg_rating_similarity'] for u, v, d in edges]
    
    # Draw ALL edges (don't filter) to match the 15-user network density
    edge_colors = []
    edge_widths = []
    
    for u, v, d in edges:
        sim = d['avg_rating_similarity']
        shared = d['shared_restaurants']
        
        # Color by similarity like the 15-user network
        if sim >= 4.0:
            edge_colors.append('green')
        elif sim >= 3.0:
            edge_colors.append('orange')
        else:
            edge_colors.append('red')
        
        # Width by shared restaurants (normalized)
        edge_widths.append(shared * 1.5 + 1)  # 1-4 range
    
    # Draw all edges
    nx.draw_networkx_edges(user_graph, pos, width=edge_widths,
                          edge_color=edge_colors, alpha=0.7)
    
    # Node sizes based on degree (larger range for better visibility)
    degrees = dict(user_graph.degree())
    node_sizes = [degrees[n] * 30 + 300 for n in user_graph.nodes()]
    
    # Draw nodes with consistent styling
    nx.draw_networkx_nodes(user_graph, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, 
                          edgecolors='navy', linewidths=2)
    
    # Add labels for ALL users (like the 15-user network)
    labels = {}
    for node in user_graph.nodes():
        name = B.nodes[node]['name']
        # Clean up user names for display
        if name.startswith('User_'):
            labels[node] = f"U{name[5:10]}"
        elif len(name) > 8:
            labels[node] = name[:8]
        else:
            labels[node] = name
    
    # Draw all labels
    nx.draw_networkx_labels(user_graph, pos, labels=labels, 
                           font_size=9, font_weight='bold', font_color='darkblue')
    
    # Create legend identical to 15-user network
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=12, markeredgecolor='navy', markeredgewidth=2, 
               label='Users', linestyle='None'),
        Line2D([0], [0], color='green', linewidth=3, 
               label='Very Similar Tastes (4+ similarity)'),
        Line2D([0], [0], color='orange', linewidth=2, 
               label='Somewhat Similar (3-4 similarity)'),
        Line2D([0], [0], color='red', linewidth=1, 
               label='Different Tastes (<3 similarity)'),
        Line2D([0], [0], color='black', linewidth=1, 
               label='Line thickness = # shared restaurants')
    ]
    
    plt.title('User Projection Network: 30 Users Connected by Restaurant Reviews\n'
              f'{user_graph.number_of_nodes()} users, {user_graph.number_of_edges()} connections\n'
              f'Network Density: {nx.density(user_graph):.3f}', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Extended user network visualization saved as '{output_file}'")


def main():
    """
    Main function for improved extended user projection analysis
    """
    print("="*70)
    print("IMPROVED EXTENDED USER PROJECTION NETWORK - 30 USERS")
    print("="*70)
    
    # Create extended sample with better connectivity
    restaurants_df, reviews_df, users_df = create_extended_sample()
    
    # Create bipartite graph
    B = create_bipartite_graph(reviews_df, restaurants_df, users_df)
    
    # Project to user network
    user_graph = project_to_user_network(B)
    
    # Analyze the network
    analyze_extended_user_network(user_graph, B)
    
    # Create visualization
    print("\n" + "="*50)
    print("CREATING IMPROVED VISUALIZATION")
    print("="*50)
    
    visualize_extended_user_network(user_graph, B, 'extended_user_network_improved.png')
    
    # Save network file
    nx.write_gexf(user_graph, "extended_user_network_improved.gexf")
    print("Improved extended user network saved as 'extended_user_network_improved.gexf'")
    
    print("\n" + "="*70)
    print("IMPROVED ANALYSIS COMPLETE")
    print("="*70)
    print("Files created:")
    print("- extended_user_network_improved.png (improved 30-user visualization)")
    print("- extended_user_network_improved.gexf (network file)")
    print("- Extended sample CSV files updated")
    
    return user_graph


if __name__ == "__main__":
    main()