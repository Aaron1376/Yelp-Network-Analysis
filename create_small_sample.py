"""
Create small sample bipartite graph with 10 users and 5 businesses

This script creates a very small dataset for demonstration and testing purposes.
"""

import pandas as pd
import numpy as np


def create_small_sample():
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
    
    # Print restaurant details
    print(f"\nSelected restaurants:")
    for _, restaurant in small_restaurants_df.iterrows():
        review_count = len(final_reviews_df[final_reviews_df['business_id'] == restaurant['business_id']])
        print(f"- {restaurant['name'][:40]:<40} ({review_count} reviews)")
    
    # Print user details
    print(f"\nSelected users:")
    for _, user in small_users_df.iterrows():
        review_count = len(final_reviews_df[final_reviews_df['user_id'] == user['user_id']])
        user_name = user.get('name', f"User_{user['user_id'][:8]}")
        print(f"- {user_name[:20]:<20} ({review_count} reviews)")
    
    return small_restaurants_df, small_users_df, final_reviews_df


if __name__ == "__main__":
    print("=" * 60)
    print("CREATING SMALL SAMPLE FOR BIPARTITE GRAPH")
    print("=" * 60)
    
    create_small_sample()
    
    print("\n" + "="*60)
    print("SMALL SAMPLE CREATION COMPLETE")
    print("="*60)
    print("Files created:")
    print("- small_restaurants_sample.csv (10 restaurants)")
    print("- small_users_sample.csv (15 users)")
    print("- small_reviews_sample.csv (reviews connecting them)")