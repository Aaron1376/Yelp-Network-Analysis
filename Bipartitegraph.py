import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def load_sample_data():
    """
    Load the sample Santa Barbara restaurants, reviews, and users data
    """
    restaurants_df = pd.read_csv('santa_barbara_restaurants_sample.csv')
    reviews_df = pd.read_csv('santa_barbara_reviews_sample.csv')
    users_df = pd.read_csv('santa_barbara_users_sample.csv')
    return restaurants_df, reviews_df, users_df

def create_small_sample_from_data():
    """
    Create a small sample with 10 businesses and 15 users from the 500x500 sample
    """
    print("Creating small sample: 10 businesses and 15 users from 500x500 sample")
    
    # Read the full sample data (500x500)
    restaurants_df = pd.read_csv('santa_barbara_restaurants_sample.csv')
    reviews_df = pd.read_csv('santa_barbara_reviews_sample.csv')
    users_df = pd.read_csv('santa_barbara_users_sample.csv')
    
    print(f"Available data: {len(restaurants_df)} restaurants, {len(users_df)} users, {len(reviews_df)} reviews")
    
    # Select 10 restaurants with the most reviews
    restaurant_review_counts = reviews_df.groupby('business_id').size().reset_index(name='review_count')
    top_restaurants = restaurant_review_counts.nlargest(10, 'review_count')
    selected_restaurant_ids = set(top_restaurants['business_id'])
    
    # Filter restaurants
    small_restaurants_df = restaurants_df[restaurants_df['business_id'].isin(selected_restaurant_ids)]
    
    # Get reviews for these restaurants
    small_reviews_df = reviews_df[reviews_df['business_id'].isin(selected_restaurant_ids)]
    
    # Select 15 users who reviewed these restaurants
    user_review_counts = small_reviews_df.groupby('user_id').size().reset_index(name='review_count')
    top_users = user_review_counts.nlargest(15, 'review_count')
    selected_user_ids = set(top_users['user_id'])
    
    # Filter users
    small_users_df = users_df[users_df['user_id'].isin(selected_user_ids)]
    
    # Filter reviews to only include selected users and restaurants
    final_reviews_df = small_reviews_df[
        (small_reviews_df['user_id'].isin(selected_user_ids)) &
        (small_reviews_df['business_id'].isin(selected_restaurant_ids))
    ]
    
    # Save small sample files
    small_restaurants_df.to_csv('small_restaurants_sample.csv', index=False)
    small_users_df.to_csv('small_users_sample.csv', index=False)
    final_reviews_df.to_csv('small_reviews_sample.csv', index=False)
    
    print(f"\nSmall sample created:")
    print(f"- Restaurants: {len(small_restaurants_df)}")
    print(f"- Users: {len(small_users_df)}")
    print(f"- Reviews: {len(final_reviews_df)}")
    
    return small_restaurants_df, final_reviews_df, small_users_df

def create_bipartite_graph(reviews_df, restaurants_df, users_df):
    """
    Create a bipartite graph connecting users to restaurants they reviewed
    """
    # Create a bipartite graph
    B = nx.Graph()
    
    # Add nodes with bipartite attribute
    # Add restaurant nodes (top set)
    restaurant_names = dict(zip(restaurants_df['business_id'], restaurants_df['name']))
    for business_id in restaurants_df['business_id']:
        B.add_node(f"R_{business_id}", bipartite=0, type='restaurant', 
                  name=restaurant_names.get(business_id, "Unknown"))
    
    # Add user nodes (bottom set)
    user_names = dict(zip(users_df['user_id'], users_df.get('name', users_df['user_id'])))
    for user_id in users_df['user_id']:
        B.add_node(f"U_{user_id}", bipartite=1, type='user', 
                  name=user_names.get(user_id, f"User_{user_id[:8]}"))
    
    # Add edges based on reviews
    for _, review in reviews_df.iterrows():
        business_id = review['business_id']
        user_id = review['user_id']
        rating = review.get('stars', 0)
        
        # Create edge between user and restaurant
        B.add_edge(f"U_{user_id}", f"R_{business_id}", 
                  weight=1, rating=rating)
    
    return B

def analyze_bipartite_network(B):
    """
    Analyze the bipartite network and print statistics
    """
    print("\nBipartite Network Statistics:")
    print(f"Total nodes: {B.number_of_nodes():,}")
    print(f"Total edges: {B.number_of_edges():,}")
    
    # Separate nodes by type
    restaurant_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'restaurant'}
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    
    print(f"Restaurant nodes: {len(restaurant_nodes):,}")
    print(f"User nodes: {len(user_nodes):,}")
    
    # Check if graph is actually bipartite
    is_bipartite = nx.is_bipartite(B)
    print(f"Is bipartite: {is_bipartite}")
    
    if B.number_of_nodes() > 0:
        # Calculate degree statistics for each type
        restaurant_degrees = [B.degree(n) for n in restaurant_nodes]
        user_degrees = [B.degree(n) for n in user_nodes]
        
        if restaurant_degrees:
            avg_restaurant_degree = sum(restaurant_degrees) / len(restaurant_degrees)
            print(f"Average reviews per restaurant: {avg_restaurant_degree:.2f}")
            
            # Top restaurants by number of reviews
            restaurant_degree_dict = {n: B.degree(n) for n in restaurant_nodes}
            top_restaurants = sorted(restaurant_degree_dict.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 restaurants by number of reviews:")
            for rest_node, degree in top_restaurants:
                name = B.nodes[rest_node]['name']
                print(f"  {name}: {degree} reviews")
        
        if user_degrees:
            avg_user_degree = sum(user_degrees) / len(user_degrees)
            print(f"Average restaurants reviewed per user: {avg_user_degree:.2f}")
            
            # Top users by number of reviews
            user_degree_dict = {n: B.degree(n) for n in user_nodes}
            top_users = sorted(user_degree_dict.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 users by number of reviews:")
            for user_node, degree in top_users:
                name = B.nodes[user_node]['name']
                print(f"  {name}: {degree} reviews")

def visualize_small_bipartite_network(B, output_file='small_bipartite_network.png'):
    """
    Create and save a detailed visualization of the small bipartite network (15 users × 10 businesses)
    """
    plt.figure(figsize=(20, 12))
    
    # Separate nodes by type
    restaurant_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'restaurant'}
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    
    # Create a bipartite layout with more spacing
    pos = {}
    
    # Position restaurants on the left in a vertical column
    restaurant_list = list(restaurant_nodes)
    restaurant_y_spacing = 2.0
    for i, node in enumerate(restaurant_list):
        pos[node] = (0, i * restaurant_y_spacing)
    
    # Position users on the right in a vertical column
    user_list = list(user_nodes)
    user_y_spacing = len(restaurant_list) * restaurant_y_spacing / len(user_list) if len(user_list) > 0 else 1.0
    for i, node in enumerate(user_list):
        pos[node] = (5, i * user_y_spacing)
    
    # Draw restaurant nodes (left side) - larger and more distinct
    nx.draw_networkx_nodes(B, pos, nodelist=restaurant_nodes, 
                          node_color='lightcoral', node_size=800, 
                          node_shape='s', label='Restaurants')
    
    # Draw user nodes (right side) - smaller circles
    nx.draw_networkx_nodes(B, pos, nodelist=user_nodes, 
                          node_color='lightblue', node_size=300, 
                          node_shape='o', label='Users')
    
    # Draw edges with varying thickness based on rating
    for edge in B.edges(data=True):
        rating = edge[2].get('rating', 3.0)
        # Thicker lines for higher ratings
        width = rating / 2.0
        alpha = 0.6
        color = 'green' if rating >= 4.0 else 'orange' if rating >= 3.0 else 'red'
        nx.draw_networkx_edges(B, pos, edgelist=[edge[:2]], width=width, 
                              alpha=alpha, edge_color=color)
    
    # Add detailed labels for all nodes
    restaurant_labels = {}
    for node in restaurant_nodes:
        name = B.nodes[node]['name']
        # Truncate long names but keep them readable
        if len(name) > 25:
            name = name[:22] + "..."
        restaurant_labels[node] = name
    
    user_labels = {}
    for node in user_nodes:
        name = B.nodes[node]['name']
        # Clean up user names
        if name.startswith('User_'):
            name = f"U{name[5:13]}"  # Use first 8 chars after 'User_'
        elif len(name) > 15:
            name = name[:12] + "..."
        user_labels[node] = name
    
    # Draw labels with better positioning
    nx.draw_networkx_labels(B, pos, labels=restaurant_labels, 
                          font_size=10, font_color='darkred', font_weight='bold')
    nx.draw_networkx_labels(B, pos, labels=user_labels, 
                          font_size=8, font_color='darkblue')
    
    # Create custom legend for edge colors
    from matplotlib.lines import Line2D
    
    # Create legend elements for edge colors
    legend_elements = [
        Line2D([0], [0], color='lightcoral', marker='s', linestyle='None', 
               markersize=10, label='Restaurants'),
        Line2D([0], [0], color='lightblue', marker='o', linestyle='None', 
               markersize=8, label='Users'),
        Line2D([0], [0], color='green', linewidth=3, label='Excellent (4-5 stars)'),
        Line2D([0], [0], color='orange', linewidth=2, label='Good (3 stars)'),
        Line2D([0], [0], color='red', linewidth=1, label='Poor (1-2 stars)')
    ]
    
    plt.title(f"Small Sample Bipartite Graph: {len(user_nodes)} Users × {len(restaurant_nodes)} Restaurants\n"
              f"Total Connections: {B.number_of_edges()}", 
              fontsize=16, fontweight='bold')
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, frameon=True, fancybox=True, shadow=True)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSmall bipartite network visualization saved as '{output_file}'")

def create_adjacency_matrix_visualization(B, output_file='small_bipartite_adjacency.png'):
    """
    Create an adjacency matrix visualization for the small bipartite graph
    """
    # Separate nodes by type
    restaurant_nodes = sorted({n for n, d in B.nodes(data=True) if d['type'] == 'restaurant'})
    user_nodes = sorted({n for n, d in B.nodes(data=True) if d['type'] == 'user'})
    
    # Create adjacency matrix
    matrix = []
    for user in user_nodes:
        row = []
        for restaurant in restaurant_nodes:
            if B.has_edge(user, restaurant):
                rating = B.edges[user, restaurant].get('rating', 0)
                row.append(rating)
            else:
                row.append(0)
        matrix.append(row)
    
    # Convert to numpy array for visualization
    import numpy as np
    matrix = np.array(matrix)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=5)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Rating (0 = No Review)', rotation=270, labelpad=15)
    
    # Set ticks and labels
    plt.xticks(range(len(restaurant_nodes)), 
               [B.nodes[r]['name'][:20] for r in restaurant_nodes], 
               rotation=45, ha='right')
    plt.yticks(range(len(user_nodes)), 
               [B.nodes[u]['name'][:15] for u in user_nodes])
    
    # Add values to the heatmap
    for i in range(len(user_nodes)):
        for j in range(len(restaurant_nodes)):
            if matrix[i, j] > 0:
                plt.text(j, i, f'{matrix[i, j]:.1f}', 
                        ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('User-Restaurant Review Matrix\n(15 Users × 10 Restaurants)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Restaurants')
    plt.ylabel('Users')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Adjacency matrix visualization saved as '{output_file}'")
    """
    Create and save a visualization of the bipartite network
    """
    plt.figure(figsize=(16, 12))
    
    # Separate nodes by type
    restaurant_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'restaurant'}
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    
    # Create a bipartite layout
    pos = {}
    
    # Position restaurants on the left
    restaurant_list = list(restaurant_nodes)
    for i, node in enumerate(restaurant_list):
        pos[node] = (0, i)
    
    # Position users on the right
    user_list = list(user_nodes)
    for i, node in enumerate(user_list):
        pos[node] = (2, i * len(restaurant_list) / len(user_list))
    
    # Draw restaurant nodes (left side)
    nx.draw_networkx_nodes(B, pos, nodelist=restaurant_nodes, 
                          node_color='lightcoral', node_size=100, 
                          label='Restaurants')
    
    # Draw user nodes (right side)
    nx.draw_networkx_nodes(B, pos, nodelist=user_nodes, 
                          node_color='lightblue', node_size=50, 
                          label='Users')
    
    # Draw edges
    nx.draw_networkx_edges(B, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Add labels for a subset of nodes (to avoid clutter)
    if len(restaurant_nodes) <= 20:
        restaurant_labels = {n: B.nodes[n]['name'][:15] + "..." 
                           if len(B.nodes[n]['name']) > 15 
                           else B.nodes[n]['name'] 
                           for n in restaurant_nodes}
        nx.draw_networkx_labels(B, pos, labels=restaurant_labels, 
                              font_size=8, font_color='darkred')
    
    plt.title("Bipartite Graph: Users Connected to Restaurants\n"
              f"({len(user_nodes)} users, {len(restaurant_nodes)} restaurants, {B.number_of_edges()} connections)", 
              fontsize=14)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBipartite network visualization saved as '{output_file}'")

def analyze_small_bipartite_sample():
    """
    Create and analyze the small bipartite sample (15 users × 10 businesses)
    """
    print("=" * 60)
    print("SMALL SAMPLE BIPARTITE GRAPH ANALYSIS (15 USERS × 10 BUSINESSES)")
    print("=" * 60)
    
    # Load or create the small sample data
    print("Loading small sample data...")
    restaurants_df, reviews_df, users_df = create_small_sample_from_data()
    
    print(f"Loaded {len(restaurants_df):,} restaurants")
    print(f"Loaded {len(reviews_df):,} reviews")
    print(f"Loaded {len(users_df):,} users")
    
    # Create the bipartite graph for small sample
    print("\nCreating small bipartite graph...")
    B_small = create_bipartite_graph(reviews_df, restaurants_df, users_df)
    
    # Analyze the small bipartite network
    analyze_bipartite_network(B_small)
    
    # Create detailed visualizations for the small sample
    print("\nCreating detailed visualizations for small sample...")
    visualize_small_bipartite_network(B_small, 'small_bipartite_detailed.png')
    create_adjacency_matrix_visualization(B_small, 'small_bipartite_matrix.png')
    
    # Save the small bipartite network
    nx.write_gexf(B_small, "small_bipartite_graph.gexf")
    print("Small bipartite graph saved as 'small_bipartite_graph.gexf'")
    
    return B_small

def main():
    """
    Main function to create and analyze bipartite networks
    """
    import sys
    
    print("Select analysis type:")
    print("1. Full sample analysis (500 users × 500 restaurants)")
    print("2. Small sample analysis (15 users × 10 restaurants)")
    print("3. Both analyses")
    
    # Check if command line argument provided
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        # Full sample analysis
        print("\n" + "=" * 60)
        print("FULL SAMPLE BIPARTITE GRAPH ANALYSIS")
        print("=" * 60)
        
        # Load the sample data
        print("Loading full sample data...")
        restaurants_df, reviews_df, users_df = load_sample_data()
        
        print(f"Loaded {len(restaurants_df):,} restaurants")
        print(f"Loaded {len(reviews_df):,} reviews")
        print(f"Loaded {len(users_df):,} users")
        
        # Create the bipartite graph
        print("\nCreating bipartite graph...")
        B = create_bipartite_graph(reviews_df, restaurants_df, users_df)
        
        # Analyze the bipartite network
        analyze_bipartite_network(B)
        
        # Save network files
        print("\nSaving full sample network files...")
        nx.write_gexf(B, "bipartite_sample.gexf")
        print("Full sample network files saved")
    
    if choice in ['2', '3']:
        # Small sample analysis
        if choice == '3':
            print("\n" + "="*80)
        
        B_small = analyze_small_bipartite_sample()
        
        # Additional small sample analysis
        print("\n" + "="*50)
        print("SMALL SAMPLE DETAILED ANALYSIS")
        print("="*50)
        
        # Analyze review patterns for small sample
        small_restaurants_df, small_reviews_df, small_users_df = create_small_sample_from_data()
        
        ratings = small_reviews_df['stars'].tolist() if 'stars' in small_reviews_df.columns else []
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            print(f"\nAverage rating across all reviews: {avg_rating:.2f}")
            
            # Rating distribution
            rating_counts = {}
            for rating in ratings:
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            
            print("Rating distribution:")
            for rating in sorted(rating_counts.keys()):
                count = rating_counts[rating]
                percentage = (count / len(ratings)) * 100
                print(f"  {rating} stars: {count} reviews ({percentage:.1f}%)")
        
        print(f"\nSmall sample files created:")
        print("- small_bipartite_detailed.png (detailed visualization)")
        print("- small_bipartite_matrix.png (adjacency matrix)")
        print("- small_bipartite_graph.gexf (network file)")
    
    if choice not in ['1', '2', '3']:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
        return None
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
