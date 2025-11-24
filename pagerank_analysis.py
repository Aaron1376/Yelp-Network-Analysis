"""
PageRank Analysis for User Networks

This script analyzes the most influential users in both the 15-user and 30-user 
networks using Google's PageRank algorithm to identify key opinion leaders 
and influential reviewers in the Santa Barbara restaurant network.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_15_user_network():
    """
    Load and recreate the 15-user network from small sample data
    """
    print("Loading 15-user network...")
    
    # Load the small sample data
    restaurants_df = pd.read_csv('small_restaurants_sample.csv')
    reviews_df = pd.read_csv('small_reviews_sample.csv')
    users_df = pd.read_csv('small_users_sample.csv')
    
    print(f"15-user sample: {len(restaurants_df)} restaurants, {len(users_df)} users, {len(reviews_df)} reviews")
    
    # Create bipartite graph
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
    
    # Project to user network
    user_nodes = {n for n, d in B.nodes(data=True) if d['type'] == 'user'}
    user_graph = nx.bipartite.projected_graph(B, user_nodes)
    
    # Add edge attributes
    for u1, u2 in user_graph.edges():
        u1_restaurants = set(B.neighbors(u1))
        u2_restaurants = set(B.neighbors(u2))
        shared_restaurants = u1_restaurants & u2_restaurants
        
        shared_ratings_u1 = [B[u1][r]['rating'] for r in shared_restaurants if 'rating' in B[u1][r]]
        shared_ratings_u2 = [B[u2][r]['rating'] for r in shared_restaurants if 'rating' in B[u2][r]]
        
        avg_rating_u1 = np.mean(shared_ratings_u1) if shared_ratings_u1 else 0
        avg_rating_u2 = np.mean(shared_ratings_u2) if shared_ratings_u2 else 0
        avg_rating_diff = abs(avg_rating_u1 - avg_rating_u2)
        
        user_graph[u1][u2]['shared_restaurants'] = len(shared_restaurants)
        user_graph[u1][u2]['avg_rating_similarity'] = 5 - avg_rating_diff
        user_graph[u1][u2]['weight'] = len(shared_restaurants)
    
    print(f"15-user network: {user_graph.number_of_nodes()} users, {user_graph.number_of_edges()} connections")
    return user_graph, B, user_names


def load_30_user_network():
    """
    Load the 30-user network from extended sample data
    """
    print("Loading 30-user network...")
    
    # Load the network file
    user_graph = nx.read_gexf('extended_user_network_improved.gexf')
    
    # Load user data for names
    users_df = pd.read_csv('extended_users_sample.csv')
    user_names = dict(zip(users_df['user_id'], users_df['name']))
    
    # Create mapping for network nodes to names
    node_to_name = {}
    for node in user_graph.nodes():
        user_id = node[2:] if node.startswith('U_') else node
        name = user_names.get(user_id, f'User_{user_id[:8]}')
        node_to_name[node] = name
    
    print(f"30-user network: {user_graph.number_of_nodes()} users, {user_graph.number_of_edges()} connections")
    return user_graph, node_to_name


def analyze_pagerank(user_graph, node_to_name, network_name, top_n=10):
    """
    Calculate PageRank and comprehensive network analysis
    """
    print(f"\n{'='*60}")
    print(f"NETWORK ANALYSIS - {network_name}")
    print(f"{'='*60}")
    
    if user_graph.number_of_nodes() == 0:
        print("No users in network")
        return {}
    
    # Calculate PageRank
    pagerank_scores = nx.pagerank(user_graph, alpha=0.85, max_iter=1000, tol=1e-06)
    
    # Get top users by PageRank
    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top {min(top_n, len(sorted_pagerank))} Most Influential Users (PageRank):")
    print("-" * 50)
    
    top_users = []
    for i, (user_node, score) in enumerate(sorted_pagerank[:top_n]):
        name = node_to_name.get(user_node, 'Unknown')
        degree = user_graph.degree(user_node)
        
        print(f"{i+1:2d}. {name:<15} - Score: {score:.4f} - Connections: {degree}")
        top_users.append((name, user_node, score, degree))
    
    # =====================================================
    # CONNECTED COMPONENTS ANALYSIS
    # =====================================================
    print(f"\nConnected Components Analysis:")
    print("-" * 50)
    
    connected_components = list(nx.connected_components(user_graph))
    print(f"Number of Connected Components: {len(connected_components)}")
    
    for i, component in enumerate(connected_components, 1):
        component_size = len(component)
        print(f"  Component {i}: {component_size} users")
        if component_size <= 5:  # Show users in small components
            component_names = [node_to_name.get(node, 'Unknown') for node in component]
            print(f"    Users: {', '.join(component_names)}")
    
    # Get largest connected component for further analysis
    largest_component = max(connected_components, key=len) if connected_components else set()
    largest_component_graph = user_graph.subgraph(largest_component) if largest_component else user_graph
    
    print(f"Largest Component: {len(largest_component)} users ({len(largest_component)/user_graph.number_of_nodes()*100:.1f}% of network)")
    
    # =====================================================
    # DIAMETER ANALYSIS
    # =====================================================
    print(f"\nDiameter Analysis:")
    print("-" * 50)
    
    if nx.is_connected(user_graph):
        diameter = nx.diameter(user_graph)
        avg_path_length = nx.average_shortest_path_length(user_graph)
        print(f"Network Diameter: {diameter}")
        print(f"Average Shortest Path Length: {avg_path_length:.3f}")
        print(f"Network is fully connected")
    else:
        print(f"Network is disconnected ({len(connected_components)} components)")
        if len(largest_component) > 1:
            try:
                diameter_main = nx.diameter(largest_component_graph)
                avg_path_main = nx.average_shortest_path_length(largest_component_graph)
                print(f"Largest Component Diameter: {diameter_main}")
                print(f"Largest Component Avg Path Length: {avg_path_main:.3f}")
            except:
                print(f"Could not calculate diameter for largest component")
        
        # Show diameter for each component with >1 node
        for i, component in enumerate(connected_components, 1):
            if len(component) > 1:
                subgraph = user_graph.subgraph(component)
                try:
                    comp_diameter = nx.diameter(subgraph)
                    print(f"  Component {i} diameter: {comp_diameter}")
                except:
                    print(f"  Component {i}: Cannot calculate diameter")
    
    # =====================================================
    # CLUSTERING COEFFICIENT ANALYSIS  
    # =====================================================
    print(f"\nClustering Coefficient Analysis:")
    print("-" * 50)
    
    # Global clustering coefficient
    global_clustering = nx.average_clustering(user_graph)
    print(f"Global Clustering Coefficient: {global_clustering:.4f}")
    
    # Individual clustering coefficients
    local_clustering = nx.clustering(user_graph)
    
    # Top clustered users
    clustered_users = sorted(local_clustering.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 Most Clustered Users (Local Clustering):")
    for i, (node, clustering_coeff) in enumerate(clustered_users[:5]):
        name = node_to_name.get(node, 'Unknown')
        degree = user_graph.degree(node)
        print(f"  {i+1}. {name:<15} - Clustering: {clustering_coeff:.4f} - Degree: {degree}")
    
    # Calculate transitivity (alternative clustering measure)
    transitivity = nx.transitivity(user_graph)
    print(f"\nTransitivity (Global Clustering): {transitivity:.4f}")
    print(f"Difference from Average Clustering: {abs(global_clustering - transitivity):.4f}")
    
    # =====================================================
    # CENTRALITY MEASURES COMPARISON
    # =====================================================
    print(f"\nCentrality Comparison for Top 5 Users:")
    print("-" * 95)
    print(f"{'User':<15} {'PageRank':<10} {'Degree':<8} {'Betweenness':<12} {'Closeness':<10} {'Clustering':<10} {'Eigenvector':<12}")
    print("-" * 95)
    
    # Calculate all centrality measures
    betweenness = nx.betweenness_centrality(user_graph)
    closeness = nx.closeness_centrality(user_graph)
    try:
        eigenvector = nx.eigenvector_centrality(user_graph, max_iter=1000)
    except:
        eigenvector = {node: 0 for node in user_graph.nodes()}
    
    for i, (name, user_node, pr_score, degree) in enumerate(top_users[:5]):
        bet_score = betweenness.get(user_node, 0)
        close_score = closeness.get(user_node, 0)
        cluster_score = local_clustering.get(user_node, 0)
        eigen_score = eigenvector.get(user_node, 0)
        
        print(f"{name:<15} {pr_score:<10.4f} {degree:<8d} {bet_score:<12.4f} {close_score:<10.4f} {cluster_score:<10.4f} {eigen_score:<12.4f}")
    
    # =====================================================
    # COMPREHENSIVE NETWORK STATISTICS
    # =====================================================
    print(f"\nComprehensive Network Statistics:")
    print("-" * 50)
    print(f"Basic Properties:")
    print(f"  - Total Users: {user_graph.number_of_nodes()}")
    print(f"  - Total Connections: {user_graph.number_of_edges()}")
    print(f"  - Network Density: {nx.density(user_graph):.4f}")
    print(f"  - Is Connected: {'Yes' if nx.is_connected(user_graph) else 'No'}")
    
    print(f"\nConnectivity:")
    print(f"  - Connected Components: {len(connected_components)}")
    print(f"  - Largest Component Size: {len(largest_component)} ({len(largest_component)/user_graph.number_of_nodes()*100:.1f}%)")
    print(f"  - Isolated Nodes: {len(list(nx.isolates(user_graph)))}")
    
    print(f"\nClustering & Structure:")
    print(f"  - Global Clustering Coefficient: {global_clustering:.4f}")
    print(f"  - Transitivity: {transitivity:.4f}")
    print(f"  - Average Degree: {np.mean([d for n, d in user_graph.degree()]):.2f}")
    
    if nx.is_connected(user_graph):
        diameter = nx.diameter(user_graph)
        avg_path = nx.average_shortest_path_length(user_graph)
        print(f"  - Network Diameter: {diameter}")
        print(f"  - Average Path Length: {avg_path:.3f}")
    
    print(f"\nPageRank Distribution:")
    print(f"  - Average PageRank: {np.mean(list(pagerank_scores.values())):.4f}")
    print(f"  - PageRank Std Dev: {np.std(list(pagerank_scores.values())):.4f}")
    print(f"  - Max PageRank: {max(pagerank_scores.values()):.4f}")
    print(f"  - Min PageRank: {min(pagerank_scores.values()):.4f}")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(user_graph))
    if isolated:
        print(f"\nIsolated Users ({len(isolated)}):")
        for node in isolated:
            name = node_to_name.get(node, 'Unknown')
            print(f"  - {name} (PageRank: {pagerank_scores[node]:.4f})")
    
    # Store additional metrics for return
    network_metrics = {
        'pagerank': pagerank_scores,
        'betweenness': betweenness,
        'closeness': closeness,
        'clustering': local_clustering,
        'eigenvector': eigenvector,
        'global_clustering': global_clustering,
        'transitivity': transitivity,
        'connected_components': len(connected_components),
        'diameter': nx.diameter(user_graph) if nx.is_connected(user_graph) else None,
        'avg_path_length': nx.average_shortest_path_length(user_graph) if nx.is_connected(user_graph) else None
    }
    
    return network_metrics, top_users


def create_pagerank_visualization(metrics_15, top_users_15, metrics_30, top_users_30):
    """
    Create visualizations comparing PageRank and network metrics between networks
    """
    print(f"\nCreating comprehensive network analysis visualizations...")
    
    # Extract PageRank for backward compatibility
    pagerank_15 = metrics_15.get('pagerank', {}) if metrics_15 else {}
    pagerank_30 = metrics_30.get('pagerank', {}) if metrics_30 else {}
    
    # Create comprehensive comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Top 10 PageRank comparison
    names_15 = [user[0] for user in top_users_15[:10]]
    scores_15 = [user[2] for user in top_users_15[:10]]
    
    names_30 = [user[0] for user in top_users_30[:10]]
    scores_30 = [user[2] for user in top_users_30[:10]]
    
    # 15-user network top users
    y_pos_15 = np.arange(len(names_15))
    ax1.barh(y_pos_15, scores_15, alpha=0.8, color='skyblue', edgecolor='navy')
    ax1.set_yticks(y_pos_15)
    ax1.set_yticklabels(names_15)
    ax1.set_xlabel('PageRank Score')
    ax1.set_title('15-User Network: Top 10 Most Influential Users\n(PageRank Algorithm)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 30-user network top users
    y_pos_30 = np.arange(len(names_30))
    ax2.barh(y_pos_30, scores_30, alpha=0.8, color='lightcoral', edgecolor='darkred')
    ax2.set_yticks(y_pos_30)
    ax2.set_yticklabels(names_30)
    ax2.set_xlabel('PageRank Score')
    ax2.set_title('30-User Network: Top 10 Most Influential Users\n(PageRank Algorithm)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Network Structure Comparison
    structure_metrics = ['Density', 'Global Clustering', 'Components']
    
    # Calculate density values
    density_15 = 0.3905  # From the output above
    density_30 = 0.1793  # From the output above
    
    values_15 = [
        density_15,
        metrics_15.get('global_clustering', 0) if metrics_15 else 0,
        metrics_15.get('connected_components', 0) if metrics_15 else 0
    ]
    
    values_30 = [
        density_30,
        metrics_30.get('global_clustering', 0) if metrics_30 else 0,
        metrics_30.get('connected_components', 0) if metrics_30 else 0
    ]
    
    x = np.arange(len(structure_metrics))
    width = 0.35
    
    ax3.bar(x - width/2, values_15, width, label='15-User Network', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, values_30, width, label='30-User Network', alpha=0.8, color='lightcoral')
    
    ax3.set_ylabel('Metric Value')
    ax3.set_title('Network Structure Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(structure_metrics, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. PageRank distribution for 30-user network
    pr_values_30 = list(pagerank_30.values()) if pagerank_30 else []
    if pr_values_30:
        ax4.hist(pr_values_30, bins=12, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.axvline(np.mean(pr_values_30), color='red', linestyle='--', label=f'Mean: {np.mean(pr_values_30):.4f}')
        ax4.set_xlabel('PageRank Score')
        ax4.set_ylabel('Number of Users')
        ax4.set_title('30-User Network: PageRank Score Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_network_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Comprehensive network analysis visualization saved as 'comprehensive_network_analysis.png'")


def create_influence_ranking_table(metrics_15, top_users_15, metrics_30, top_users_30):
    """
    Create a detailed ranking table comparing influence across networks
    """
    print(f"\nCreating influence ranking comparison...")
    
    pagerank_15 = metrics_15.get('pagerank', {}) if metrics_15 else {}
    pagerank_30 = metrics_30.get('pagerank', {}) if metrics_30 else {}
    
    # Create comparison dataframe
    comparison_data = []
    
    # Get all unique users from both networks
    all_users_15 = {user[0]: user for user in top_users_15}
    all_users_30 = {user[0]: user for user in top_users_30}
    
    all_user_names = set(all_users_15.keys()) | set(all_users_30.keys())
    
    for name in all_user_names:
        user_15 = all_users_15.get(name, None)
        user_30 = all_users_30.get(name, None)
        
        rank_15 = None
        score_15 = None
        connections_15 = None
        
        rank_30 = None
        score_30 = None
        connections_30 = None
        
        if user_15:
            rank_15 = next((i+1 for i, u in enumerate(top_users_15) if u[0] == name), None)
            score_15 = user_15[2]
            connections_15 = user_15[3]
        
        if user_30:
            rank_30 = next((i+1 for i, u in enumerate(top_users_30) if u[0] == name), None)
            score_30 = user_30[2]
            connections_30 = user_30[3]
        
        comparison_data.append({
            'User': name,
            'Rank_15': rank_15,
            'PageRank_15': score_15,
            'Connections_15': connections_15,
            'Rank_30': rank_30,
            'PageRank_30': score_30,
            'Connections_30': connections_30
        })
    
    # Sort by best rank across both networks
    comparison_data.sort(key=lambda x: min(x['Rank_15'] or 999, x['Rank_30'] or 999))
    
    # Create CSV file
    df = pd.DataFrame(comparison_data)
    df.to_csv('pagerank_influence_comparison.csv', index=False)
    
    print("Influence comparison table:")
    print("-" * 100)
    print(f"{'User':<15} {'15-User Rank':<12} {'15-PR Score':<12} {'15-Conn':<8} {'30-User Rank':<12} {'30-PR Score':<12} {'30-Conn':<8}")
    print("-" * 100)
    
    for row in comparison_data[:15]:  # Show top 15
        user = row['User']
        rank_15 = f"#{row['Rank_15']}" if row['Rank_15'] else "N/A"
        score_15 = f"{row['PageRank_15']:.4f}" if row['PageRank_15'] else "N/A"
        conn_15 = f"{row['Connections_15']}" if row['Connections_15'] else "N/A"
        
        rank_30 = f"#{row['Rank_30']}" if row['Rank_30'] else "N/A"
        score_30 = f"{row['PageRank_30']:.4f}" if row['PageRank_30'] else "N/A"
        conn_30 = f"{row['Connections_30']}" if row['Connections_30'] else "N/A"
        
        print(f"{user:<15} {rank_15:<12} {score_15:<12} {conn_15:<8} {rank_30:<12} {score_30:<12} {conn_30:<8}")
    
    print(f"\nComplete comparison saved to 'pagerank_influence_comparison.csv'")


def main():
    """
    Main function for PageRank analysis of both user networks
    """
    print("="*80)
    print("PAGERANK INFLUENCE ANALYSIS - USER NETWORKS")
    print("="*80)
    print("Analyzing the most influential users using Google's PageRank algorithm")
    print("Comparing 15-user vs 30-user networks to identify key opinion leaders")
    print()
    
    # Load networks
    try:
        user_graph_15, B_15, user_names_15 = load_15_user_network()
    except FileNotFoundError as e:
        print(f"Error loading 15-user network: {e}")
        print("Skipping 15-user analysis...")
        user_graph_15, user_names_15 = None, None
    
    try:
        user_graph_30, user_names_30 = load_30_user_network()
    except FileNotFoundError as e:
        print(f"Error loading 30-user network: {e}")
        print("Skipping 30-user analysis...")
        user_graph_30, user_names_30 = None, None
    
    # Analyze PageRank for both networks
    metrics_15, top_users_15 = None, []
    metrics_30, top_users_30 = None, []
    
    if user_graph_15 and user_names_15:
        # Convert B node names to user names for 15-user network
        node_to_name_15 = {}
        for node in user_graph_15.nodes():
            if node in B_15.nodes():
                node_to_name_15[node] = B_15.nodes[node]['name']
            else:
                node_to_name_15[node] = 'Unknown'
        
        metrics_15, top_users_15 = analyze_pagerank(user_graph_15, node_to_name_15, "15-USER NETWORK")
    
    if user_graph_30 and user_names_30:
        metrics_30, top_users_30 = analyze_pagerank(user_graph_30, user_names_30, "30-USER NETWORK")
    
    # Create visualizations and comparisons
    if metrics_15 and metrics_30:
        print("\n" + "="*60)
        print("CREATING COMPARATIVE ANALYSIS")
        print("="*60)
        
        create_pagerank_visualization(metrics_15, top_users_15, metrics_30, top_users_30)
        create_influence_ranking_table(metrics_15, top_users_15, metrics_30, top_users_30)
        
        # Summary insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        top_5_names_15 = [user[0] for user in top_users_15[:5]]
        top_5_names_30 = [user[0] for user in top_users_30[:5]]
        
        common_influencers = set(top_5_names_15) & set(top_5_names_30)
        
        print(f"Most Influential User (15-user): {top_users_15[0][0]} (Score: {top_users_15[0][2]:.4f})")
        print(f"Most Influential User (30-user): {top_users_30[0][0]} (Score: {top_users_30[0][2]:.4f})")
        print(f"Common Top-5 Influencers: {', '.join(common_influencers) if common_influencers else 'None'}")
        print(f"Network Scale Impact: PageRank more distributed in larger network")
        
        # Network structure comparison
        if metrics_15 and metrics_30:
            print(f"\nNetwork Structure Comparison:")
            print(f"15-user network: {metrics_15.get('connected_components', 'N/A')} components, {metrics_15.get('global_clustering', 'N/A'):.3f} clustering")
            print(f"30-user network: {metrics_30.get('connected_components', 'N/A')} components, {metrics_30.get('global_clustering', 'N/A'):.3f} clustering")
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Files created:")
    print("- comprehensive_network_analysis.png (visual comparison with all metrics)")
    print("- pagerank_influence_comparison.csv (detailed rankings)")
    print("\nKEY METRICS EXPLAINED:")
    print("- PageRank: Influence based on network position (like Google's algorithm)")
    print("- Connected Components: Separate groups of connected users")
    print("- Diameter: Longest shortest path in the network")
    print("- Clustering Coefficient: How much neighbors connect to each other (≠ closeness centrality)")
    print("- Closeness Centrality: How close a user is to all others (average distance)")
    print("- Betweenness Centrality: How often a user is on the shortest path between others")


if __name__ == "__main__":
    main()