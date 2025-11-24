"""
Create random samples of restaurants and users for analysis

This script creates smaller, manageable datasets by randomly sampling
500 restaurants and 500 users from the Santa Barbara dataset.
"""

import pandas as pd
import numpy as np


def create_restaurant_sample(input_file='santa_barbara_restaurants.csv', 
                           output_file='santa_barbara_restaurants_sample.csv',
                           sample_size=500):
    """
    Create a random sample of restaurants from the full dataset
    
    Args:
        input_file (str): Path to the full restaurants CSV file
        output_file (str): Path to save the sample CSV file
        sample_size (int): Number of restaurants to sample
        
    Returns:
        pd.DataFrame: Sampled restaurants dataframe
    """
    try:
        # Read the full restaurants dataset
        print(f"Reading restaurant data from: {input_file}")
        restaurants_df = pd.read_csv(input_file)
        print(f"Total restaurants available: {len(restaurants_df)}")
        
        # Check if we have enough restaurants
        if len(restaurants_df) < sample_size:
            print(f"Warning: Only {len(restaurants_df)} restaurants available, using all of them")
            sample_size = len(restaurants_df)
        
        # Create random sample
        np.random.seed(42)  # For reproducible results
        sampled_restaurants = restaurants_df.sample(n=sample_size, random_state=42)
        
        # Save to new CSV
        sampled_restaurants.to_csv(output_file, index=False)
        print(f"Sampled {len(sampled_restaurants)} restaurants saved to: {output_file}")
        
        # Print some statistics
        print(f"\nSample Restaurant Statistics:")
        if 'stars' in sampled_restaurants.columns:
            print(f"Average rating: {sampled_restaurants['stars'].mean():.2f}")
        if 'review_count' in sampled_restaurants.columns:
            print(f"Average reviews per restaurant: {sampled_restaurants['review_count'].mean():.1f}")
            print(f"Total reviews: {sampled_restaurants['review_count'].sum()}")
        
        return sampled_restaurants
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error creating restaurant sample: {e}")
        return pd.DataFrame()


def create_user_sample(restaurants_sample_df, 
                      reviews_file='santa_barbara_restaurant_reviews.csv',
                      users_file='santa_barbara_restaurant_users.csv',
                      output_file='santa_barbara_users_sample.csv',
                      sample_size=500):
    """
    Create a random sample of users who reviewed the sampled restaurants
    
    Args:
        restaurants_sample_df (pd.DataFrame): Sampled restaurants dataframe
        reviews_file (str): Path to the reviews CSV file
        users_file (str): Path to the full users CSV file
        output_file (str): Path to save the sample users CSV file
        sample_size (int): Number of users to sample
        
    Returns:
        pd.DataFrame: Sampled users dataframe
    """
    try:
        # Get business IDs from sampled restaurants
        sampled_business_ids = set(restaurants_sample_df['business_id'])
        print(f"\nFiltering users who reviewed the {len(sampled_business_ids)} sampled restaurants...")
        
        # Read reviews and filter for sampled restaurants
        print(f"Reading reviews from: {reviews_file}")
        reviews_df = pd.read_csv(reviews_file)
        print(f"Total reviews available: {len(reviews_df)}")
        
        # Filter reviews for sampled restaurants
        relevant_reviews = reviews_df[reviews_df['business_id'].isin(sampled_business_ids)]
        print(f"Reviews for sampled restaurants: {len(relevant_reviews)}")
        
        # Get unique user IDs who reviewed these restaurants
        relevant_user_ids = set(relevant_reviews['user_id'])
        print(f"Users who reviewed sampled restaurants: {len(relevant_user_ids)}")
        
        # Read full user data
        print(f"Reading user data from: {users_file}")
        users_df = pd.read_csv(users_file)
        print(f"Total users available: {len(users_df)}")
        
        # Filter users who reviewed the sampled restaurants
        relevant_users = users_df[users_df['user_id'].isin(relevant_user_ids)]
        print(f"Relevant users found: {len(relevant_users)}")
        
        # Check if we have enough users
        if len(relevant_users) < sample_size:
            print(f"Warning: Only {len(relevant_users)} relevant users available, using all of them")
            sample_size = len(relevant_users)
        
        # Create random sample of users
        np.random.seed(42)  # For reproducible results
        sampled_users = relevant_users.sample(n=sample_size, random_state=42)
        
        # Save to new CSV
        sampled_users.to_csv(output_file, index=False)
        print(f"Sampled {len(sampled_users)} users saved to: {output_file}")
        
        # Print some statistics
        print(f"\nSample User Statistics:")
        if 'review_count' in sampled_users.columns:
            print(f"Average reviews per user: {sampled_users['review_count'].mean():.1f}")
            print(f"Total reviews by sampled users: {sampled_users['review_count'].sum()}")
        if 'average_stars' in sampled_users.columns:
            print(f"Average rating given by users: {sampled_users['average_stars'].mean():.2f}")
        
        return sampled_users
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error creating user sample: {e}")
        return pd.DataFrame()


def create_filtered_reviews_sample(restaurants_sample_df, users_sample_df,
                                 reviews_file='santa_barbara_restaurant_reviews.csv',
                                 output_file='santa_barbara_reviews_sample.csv'):
    """
    Create a sample of reviews that involve only the sampled restaurants and users
    
    Args:
        restaurants_sample_df (pd.DataFrame): Sampled restaurants dataframe
        users_sample_df (pd.DataFrame): Sampled users dataframe
        reviews_file (str): Path to the full reviews CSV file
        output_file (str): Path to save the sample reviews CSV file
        
    Returns:
        pd.DataFrame: Filtered reviews dataframe
    """
    try:
        # Get IDs from samples
        sampled_business_ids = set(restaurants_sample_df['business_id'])
        sampled_user_ids = set(users_sample_df['user_id'])
        
        print(f"\nCreating reviews sample...")
        print(f"Sampled restaurants: {len(sampled_business_ids)}")
        print(f"Sampled users: {len(sampled_user_ids)}")
        
        # Read reviews
        reviews_df = pd.read_csv(reviews_file)
        
        # Filter reviews for sampled restaurants and users
        filtered_reviews = reviews_df[
            (reviews_df['business_id'].isin(sampled_business_ids)) &
            (reviews_df['user_id'].isin(sampled_user_ids))
        ]
        
        # Save filtered reviews
        filtered_reviews.to_csv(output_file, index=False)
        print(f"Filtered {len(filtered_reviews)} reviews saved to: {output_file}")
        
        # Print statistics
        print(f"\nSample Review Statistics:")
        if 'stars' in filtered_reviews.columns:
            print(f"Average rating: {filtered_reviews['stars'].mean():.2f}")
        print(f"Reviews per restaurant: {len(filtered_reviews) / len(sampled_business_ids):.1f}")
        print(f"Reviews per user: {len(filtered_reviews) / len(sampled_user_ids):.1f}")
        
        return filtered_reviews
        
    except Exception as e:
        print(f"Error creating reviews sample: {e}")
        return pd.DataFrame()


def main():
    """
    Main function to create all sample datasets
    """
    print("=" * 60)
    print("CREATING SAMPLE DATASETS")
    print("=" * 60)
    
    # Create restaurant sample
    print("\n" + "="*50)
    print("SAMPLING RESTAURANTS")
    print("="*50)
    restaurants_sample = create_restaurant_sample(sample_size=500)
    
    if not restaurants_sample.empty:
        # Create user sample based on restaurant sample
        print("\n" + "="*50)
        print("SAMPLING USERS")
        print("="*50)
        users_sample = create_user_sample(restaurants_sample, sample_size=500)
        
        if not users_sample.empty:
            # Create filtered reviews sample
            print("\n" + "="*50)
            print("FILTERING REVIEWS")
            print("="*50)
            reviews_sample = create_filtered_reviews_sample(restaurants_sample, users_sample)
            
            print("\n" + "="*60)
            print("SAMPLING COMPLETE")
            print("="*60)
            print(f"Files created:")
            print(f"- santa_barbara_restaurants_sample.csv ({len(restaurants_sample)} restaurants)")
            print(f"- santa_barbara_users_sample.csv ({len(users_sample)} users)")
            print(f"- santa_barbara_reviews_sample.csv ({len(reviews_sample)} reviews)")
        else:
            print("Failed to create user sample")
    else:
        print("Failed to create restaurant sample")


if __name__ == "__main__":
    main()