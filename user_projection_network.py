"""
User Projection Network Analysis

This script projects the bipartite graph onto the user set, creating a user-user network
where users are connected if they reviewed the same restaurants. Uses the small sample
(15 users × 10 restaurants) for clear visualization.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def create_small_sample_from_data():
    """
    Create a small sample with 10 businesses and 15 users from the 500x500 sample
    """
    print("Loading small sample data...")
    
    try:
        # Try to load existing small sample files
        restaurants_df = pd.read_csv('small_restaurants_sample.csv')
        reviews_df = pd.read_csv('small_reviews_sample.csv')
        users_df = pd.read_csv('small_users_sample.csv')
        print("Small sample files found and loaded.")
    except FileNotFoundError:
        print("Small sample files not found. Creating them from 500x500 sample...")
        
        # Read the full sample data (500x500)
        restaurants_df = pd.read_csv('santa_barbara_restaurants_sample.csv')
        reviews_df = pd.read_csv('santa_barbara_reviews_sample.csv')
        users_df = pd.read_csv('santa_barbara_users_sample.csv')
        
        # Select 10 restaurants with the most reviews
        restaurant_review_counts = reviews_df.groupby('business_id').size().reset_index(name='review_count')
        top_restaurants = restaurant_review_counts.nlargest(10, 'review_count')
        selected_restaurant_ids = set(top_restaurants['business_id'])
        
        # Filter restaurants
        restaurants_df = restaurants_df[restaurants_df['business_id'].isin(selected_restaurant_ids)]
        
        # Get reviews for these restaurants
        small_reviews_df = reviews_df[reviews_df['business_id'].isin(selected_restaurant_ids)]
        
        # Select 15 users who reviewed these restaurants
        user_review_counts = small_reviews_df.groupby('user_id').size().reset_index(name='review_count')
        top_users = user_review_counts.nlargest(15, 'review_count')
        selected_user_ids = set(top_users['user_id'])
        
        # Filter users and reviews
        users_df = users_df[users_df['user_id'].isin(selected_user_ids)]
        reviews_df = small_reviews_df[
            (small_reviews_df['user_id'].isin(selected_user_ids)) &
            (small_reviews_df['business_id'].isin(selected_restaurant_ids))
        ]
        
        # Save small sample files
        restaurants_df.to_csv('small_restaurants_sample.csv', index=False)
        users_df.to_csv('small_users_sample.csv', index=False)
        reviews_df.to_csv('small_reviews_sample.csv', index=False)
        print("Small sample files created and saved.")
    
    print(f"Sample loaded: {len(restaurants_df)} restaurants, {len(users_df)} users, {len(reviews_df)} reviews")
    return restaurants_df, reviews_df, users_df


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
    Users are connected if they reviewed the same restaurant(s)
    """
    print("Projecting bipartite graph onto user set...")
    
    # Get user nodes
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    
    # Create projection onto user nodes
    user_graph = nx.bipartite.projected_graph(B, user_nodes)
    
    # Add additional attributes to edges (shared restaurants and average ratings)
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
        user_graph[u1][u2]['avg_rating_similarity'] = 5 - avg_rating_diff  # Higher is more similar
        user_graph[u1][u2]['weight'] = len(shared_restaurants)
    
    print(f"User projection created: {user_graph.number_of_nodes()} users, {user_graph.number_of_edges()} connections")
    return user_graph


def analyze_user_network(user_graph, B):
    """
    Analyze the user projection network
    """
    print("\n" + "="*60)
    print("USER PROJECTION NETWORK ANALYSIS")
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
        print(f"Average connections per user: {avg_degree:.2f}")
        
        # Find most connected users
        degree_dict = dict(user_graph.degree())
        top_users = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nMost connected users:")
        for user_node, degree in top_users:
            name = B.nodes[user_node]['name']
            print(f"  {name}: {degree} connections")
        
        # Analyze shared restaurant patterns
        shared_counts = [user_graph[u1][u2]['shared_restaurants'] for u1, u2 in user_graph.edges()]
        if shared_counts:
            avg_shared = np.mean(shared_counts)
            max_shared = max(shared_counts)
            print(f"\nShared restaurant statistics:")
            print(f"  Average shared restaurants per connection: {avg_shared:.2f}")
            print(f"  Maximum shared restaurants: {max_shared}")
        
        # Find users with highest rating similarity
        similarities = [user_graph[u1][u2]['avg_rating_similarity'] for u1, u2 in user_graph.edges()]
        if similarities:
            avg_similarity = np.mean(similarities)
            print(f"  Average rating similarity: {avg_similarity:.2f}/5.0")


def visualize_user_network(user_graph, B, output_file='user_projection_network.png'):
    """
    Create a detailed visualization of the user projection network
    """
    print(f"\nCreating user network visualization...")
    
    plt.figure(figsize=(16, 12))
    
    # Create layout
    pos = nx.spring_layout(user_graph, k=3, iterations=50)
    
    # Prepare edge attributes for visualization
    edges = user_graph.edges(data=True)
    edge_weights = [d['shared_restaurants'] for u, v, d in edges]
    edge_similarities = [d['avg_rating_similarity'] for u, v, d in edges]
    
    # Normalize edge weights for thickness
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [(w / max_weight) * 5 + 1 for w in edge_weights]  # 1-6 range
    
    # Color edges by rating similarity
    edge_colors = []
    for sim in edge_similarities:
        if sim >= 4.5:
            edge_colors.append('green')  # Very similar tastes
        elif sim >= 3.5:
            edge_colors.append('orange')  # Somewhat similar
        else:
            edge_colors.append('red')     # Different tastes
    
    # Draw edges
    nx.draw_networkx_edges(user_graph, pos, width=edge_widths, 
                          edge_color=edge_colors, alpha=0.6)
    
    # Node sizes based on degree (number of connections)
    node_sizes = [user_graph.degree(n) * 200 + 300 for n in user_graph.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(user_graph, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, 
                          edgecolors='navy', linewidths=2)
    
    # Add labels
    labels = {}
    for node in user_graph.nodes():
        name = B.nodes[node]['name']
        # Clean up user names for display
        if name.startswith('User_'):
            labels[node] = f"U{name[5:13]}"
        elif len(name) > 12:
            labels[node] = name[:9] + "..."
        else:
            labels[node] = name
    
    nx.draw_networkx_labels(user_graph, pos, labels=labels, 
                           font_size=10, font_weight='bold', font_color='darkblue')
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=15, markeredgecolor='navy', markeredgewidth=2, 
               label='Users', linestyle='None'),
        Line2D([0], [0], color='green', linewidth=4, 
               label='Very Similar Tastes (4.5+ similarity)'),
        Line2D([0], [0], color='orange', linewidth=3, 
               label='Somewhat Similar (3.5-4.5 similarity)'),
        Line2D([0], [0], color='red', linewidth=2, 
               label='Different Tastes (<3.5 similarity)'),
        Line2D([0], [0], color='black', linewidth=1, 
               label='Line thickness = # shared restaurants')
    ]
    
    plt.title('User Projection Network: Users Connected by Shared Restaurant Reviews\n'
              f'{user_graph.number_of_nodes()} users, {user_graph.number_of_edges()} connections\n'
              f'Network Density: {nx.density(user_graph):.3f}', 
              fontsize=14, fontweight='bold')
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"User network visualization saved as '{output_file}'")


def create_user_similarity_matrix(user_graph, B, output_file='user_similarity_matrix.png'):
    """
    Create a similarity matrix heatmap for users
    """
    print("Creating user similarity matrix...")
    
    users = list(user_graph.nodes())
    n_users = len(users)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n_users, n_users))
    
    for i, user1 in enumerate(users):
        for j, user2 in enumerate(users):
            if i == j:
                similarity_matrix[i][j] = 5.0  # Perfect similarity with self
            elif user_graph.has_edge(user1, user2):
                similarity_matrix[i][j] = user_graph[user1][user2]['avg_rating_similarity']
            else:
                similarity_matrix[i][j] = 0  # No shared restaurants
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=5, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Rating Similarity (5 = Most Similar)', rotation=270, labelpad=20)
    
    # Set labels
    user_labels = []
    for user in users:
        name = B.nodes[user]['name']
        if name.startswith('User_'):
            user_labels.append(f"U{name[5:13]}")
        elif len(name) > 10:
            user_labels.append(name[:10] + "...")
        else:
            user_labels.append(name)
    
    plt.xticks(range(n_users), user_labels, rotation=45, ha='right')
    plt.yticks(range(n_users), user_labels)
    
    # Add values to cells
    for i in range(n_users):
        for j in range(n_users):
            value = similarity_matrix[i][j]
            color = 'white' if value > 2.5 else 'black'
            plt.text(j, i, f'{value:.1f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=8)
    
    plt.title('User Rating Similarity Matrix\n'
              '(Based on shared restaurant ratings)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Users')
    plt.ylabel('Users')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"User similarity matrix saved as '{output_file}'")


def main():
    """
    Main function to create and analyze user projection network
    """
    print("="*70)
    print("USER PROJECTION NETWORK ANALYSIS")
    print("="*70)
    
    # Load small sample data
    restaurants_df, reviews_df, users_df = create_small_sample_from_data()
    
    # Create bipartite graph
    B = create_bipartite_graph(reviews_df, restaurants_df, users_df)
    
    # Project to user network
    user_graph = project_to_user_network(B)
    
    # Analyze the network
    analyze_user_network(user_graph, B)
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    visualize_user_network(user_graph, B, 'user_projection_network.png')
    create_user_similarity_matrix(user_graph, B, 'user_similarity_matrix.png')
    
    # Save network file
    nx.write_gexf(user_graph, "user_projection_network.gexf")
    print("User projection network saved as 'user_projection_network.gexf'")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Files created:")
    print("- user_projection_network.png (network visualization)")
    print("- user_similarity_matrix.png (similarity heatmap)")
    print("- user_projection_network.gexf (network file)")
    
    return user_graph


if __name__ == "__main__":
    main()