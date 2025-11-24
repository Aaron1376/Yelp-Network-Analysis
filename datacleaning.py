"""
Santa Barbara Yelp Data Processing and Analysis Tool

This module provides functions to extract, filter, and analyze Yelp business data 
specifically for the Santa Barbara area. It processes large JSON files and creates 
CSV files for easier analysis.

Author: Aaron
Date: November 2025
"""

import json
import pandas as pd
from collections import Counter


# =============================================================================
# BUSINESS DATA PROCESSING FUNCTIONS
# =============================================================================

def parse_santa_barbara_businesses(file_path):
    """
    Extract all businesses from Santa Barbara area and return as a DataFrame.
    Includes all variations of Santa Barbara naming.
    """
    businesses = []
    line_count = 0
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                try:
                    business = json.loads(line.strip())
                    # Check if business is in Santa Barbara (case-insensitive)
                    if business.get('city') and 'santa barbara' in business['city'].lower():
                        businesses.append(business)
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:  # Only show first 5 errors
                        print(f"Error parsing JSON on line {line_count}: {e}")
                    continue
                
                # Print progress every 50000 lines
                if line_count % 50000 == 0:
                    print(f"Processed {line_count} lines...")
    
        print(f"\nFinished processing {line_count} lines")
        if error_count > 5:
            print(f"(Suppressed {error_count - 5} additional error messages)")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(businesses)
        return df
    
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        return pd.DataFrame()


# =============================================================================
# REVIEW DATA PROCESSING FUNCTIONS
# =============================================================================

def filter_reviews_for_businesses(review_file_path, business_ids):
    """
    Filter reviews to only include those for specific business IDs.
    
    Args:
        review_file_path (str): Path to the reviews JSON file
        business_ids (set): Set of business IDs to filter for
        
    Returns:
        pd.DataFrame: DataFrame containing filtered reviews
    """
    reviews = []
    line_count = 0
    error_count = 0
    matched_count = 0
    
    try:
        with open(review_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                try:
                    review = json.loads(line.strip())
                    if review.get('business_id') in business_ids:
                        reviews.append(review)
                        matched_count += 1
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Error parsing JSON on line {line_count}: {e}")
                    continue
                
                # Print progress every 100000 lines
                if line_count % 100000 == 0:
                    print(f"Processed {line_count:,} reviews, found {matched_count:,} matches...")
    
        print(f"\nFinished processing {line_count:,} reviews")
        print(f"Found {matched_count:,} reviews for Santa Barbara businesses")
        
        return pd.DataFrame(reviews)
    
    except FileNotFoundError:
        print(f"Error: Could not find file '{review_file_path}'")
        return pd.DataFrame()


def filter_reviews_for_restaurants(review_file_path='yelp_academic_dataset_review.json', 
                                  restaurants_csv='santa_barbara_restaurants.csv', 
                                  output_file='santa_barbara_restaurant_reviews.csv'):
    """
    Filter reviews to only include those for restaurants in the restaurants CSV file.
    
    Args:
        review_file_path (str): Path to the reviews JSON file
        restaurants_csv (str): Path to the restaurants CSV file
        output_file (str): Path to save the filtered reviews
        
    Returns:
        pd.DataFrame: DataFrame containing filtered reviews for restaurants
    """
    try:
        # Read the restaurants CSV to get business IDs
        print(f"Reading restaurant business IDs from: {restaurants_csv}")
        restaurants_df = pd.read_csv(restaurants_csv)
        restaurant_business_ids = set(restaurants_df['business_id'])
        print(f"Found {len(restaurant_business_ids):,} restaurant business IDs")
        
        # Filter reviews for these restaurants
        print(f"Filtering reviews from: {review_file_path}")
        reviews = []
        line_count = 0
        error_count = 0
        matched_count = 0
        
        with open(review_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                try:
                    review = json.loads(line.strip())
                    if review.get('business_id') in restaurant_business_ids:
                        reviews.append(review)
                        matched_count += 1
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Error parsing JSON on line {line_count}: {e}")
                    continue
                
                # Print progress every 100000 lines
                if line_count % 100000 == 0:
                    print(f"Processed {line_count:,} reviews, found {matched_count:,} restaurant reviews...")
        
        # Convert to DataFrame and save
        reviews_df = pd.DataFrame(reviews)
        if not reviews_df.empty:
            reviews_df.to_csv(output_file, index=False)
        
        # Display results
        print(f"\n{'='*60}")
        print("RESTAURANT REVIEWS FILTERING RESULTS")
        print(f"{'='*60}")
        print(f"Total reviews processed: {line_count:,}")
        print(f"Restaurant reviews found: {matched_count:,}")
        print(f"Percentage of reviews for restaurants: {(matched_count/line_count*100):.2f}%")
        print(f"Reviews saved to: {output_file}")
        
        if error_count > 5:
            print(f"(Suppressed {error_count - 5} additional error messages)")
        
        return reviews_df
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find file - {e}")
        print("Please ensure both the review JSON file and restaurants CSV file exist.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: Failed to filter reviews - {e}")
        return pd.DataFrame()


# =============================================================================
# USER DATA PROCESSING FUNCTIONS
# =============================================================================

def filter_users_for_reviews(user_file_path, user_ids):
    """
    Filter users to only include those who wrote reviews for Santa Barbara businesses.
    
    Args:
        user_file_path (str): Path to the users JSON file
        user_ids (set): Set of user IDs to filter for
        
    Returns:
        pd.DataFrame: DataFrame containing filtered users
    """
    users = []
    line_count = 0
    error_count = 0
    matched_count = 0
    
    try:
        with open(user_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                try:
                    user = json.loads(line.strip())
                    if user.get('user_id') in user_ids:
                        users.append(user)
                        matched_count += 1
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Error parsing JSON on line {line_count}: {e}")
                    continue
                
                # Print progress every 100000 lines
                if line_count % 100000 == 0:
                    print(f"Processed {line_count:,} users, found {matched_count:,} matches...")
    
        print(f"\nFinished processing {line_count:,} users")
        print(f"Found {matched_count:,} users who reviewed Santa Barbara businesses")
        
        return pd.DataFrame(users)
    
    except FileNotFoundError:
        print(f"Error: Could not find file '{user_file_path}'")
        return pd.DataFrame()


def filter_restaurants_from_businesses(businesses_df, output_file='santa_barbara_restaurants.csv'):
    """
    Filter businesses to only include restaurants based on categories.
    
    Args:
        businesses_df (pd.DataFrame): DataFrame containing all businesses
        output_file (str): Path to save the filtered restaurants CSV
        
    Returns:
        pd.DataFrame: DataFrame containing only restaurants
    """
    if businesses_df.empty:
        print("No business data provided")
        return pd.DataFrame()
    
    # Restaurant-related keywords to search for in categories
    restaurant_keywords = [
        'restaurants', 'restaurant', 'food', 'dining',
        'mexican', 'italian', 'chinese', 'japanese', 'thai', 'vietnamese',
        'american (new)', 'american (traditional)', 'french', 'indian',
        'mediterranean', 'seafood', 'steakhouses', 'pizza', 'burgers',
        'sandwiches', 'delis', 'cafes', 'breakfast & brunch',
        'fast food', 'caterers', 'buffets', 'diners',
        'asian fusion', 'latin american', 'middle eastern',
        'vegetarian', 'vegan', 'gluten-free'
    ]
    
    restaurants = []
    
    for _, business in businesses_df.iterrows():
        categories = business.get('categories', '')
        if categories and isinstance(categories, str):
            # Convert to lowercase for case-insensitive matching
            categories_lower = categories.lower()
            
            # Check if any restaurant keyword is in the categories
            is_restaurant = any(keyword in categories_lower for keyword in restaurant_keywords)
            
            if is_restaurant:
                restaurants.append(business)
    
    restaurants_df = pd.DataFrame(restaurants)
    
    print(f"\nRestaurant Filtering Results:")
    print(f"Total businesses: {len(businesses_df):,}")
    print(f"Restaurants found: {len(restaurants_df):,}")
    print(f"Percentage of restaurants: {(len(restaurants_df)/len(businesses_df)*100):.1f}%")
    
    if not restaurants_df.empty:
        # Save to CSV
        restaurants_df.to_csv(output_file, index=False)
        print(f"Restaurants saved to: {output_file}")
        
        # Show top restaurant categories
        print(f"\nTop restaurant categories:")
        all_categories = []
        for cats in restaurants_df['categories'].dropna():
            if cats:
                all_categories.extend([c.strip() for c in cats.split(',')])
        
        category_counts = Counter(all_categories)
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            percentage = (count / len(restaurants_df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
    
    return restaurants_df


# =============================================================================
# DATA ANALYSIS AND PROCESSING FUNCTIONS
# =============================================================================

def process_business_data(df):
    """
    Process and display business statistics
    """
    print("\nSanta Barbara Business Statistics:")
    print(f"Total number of businesses: {len(df):,}")
    
    if 'categories' in df.columns:
        all_categories = []
        for cats in df['categories'].dropna():
            if cats:
                all_categories.extend([c.strip() for c in cats.split(',')])
        
        category_counts = Counter(all_categories)
        print("\nTop 20 business categories:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"{category}: {count:,} businesses")
    
    if 'review_count' in df.columns:
        print(f"\nTotal reviews: {df['review_count'].sum():,}")
        print(f"Average reviews per business: {df['review_count'].mean():.1f}")
    
    if 'stars' in df.columns:
        print(f"Average rating: {df['stars'].mean():.2f} stars")
    
    return df

def save_data_to_csv(df, filename):
    """
    Save DataFrame to CSV file
    """
    df.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")
    return filename


def process_review_data(reviews_df):
    """
    Process and display review statistics
    """
    if not reviews_df.empty:
        print(f"Total reviews saved: {len(reviews_df):,}")
        user_ids = set(reviews_df['user_id'])
        print(f"\nFound {len(user_ids):,} unique users who wrote reviews")
        return user_ids
    return set()


def process_user_data(users_df):
    """
    Process and display user statistics
    """
    if not users_df.empty:
        print(f"Total users saved: {len(users_df):,}")
        
        if 'review_count' in users_df.columns:
            print(f"\nAverage reviews per user: {users_df['review_count'].mean():.1f}")
        if 'average_stars' in users_df.columns:
            print(f"Average rating given by users: {users_df['average_stars'].mean():.2f} stars")


# =============================================================================
# RESTAURANT FILTERING FUNCTIONS
# =============================================================================

def filter_restaurants(csv_file_path='santa_barbara_businesses.csv', output_file='santa_barbara_restaurants.csv'):
    """
    Filter businesses from CSV to only include those with "Restaurants" category and save to a new CSV file.
    
    Args:
        csv_file_path (str): Path to the input CSV file with business data
        output_file (str): Path to save the filtered restaurant data
    
    Returns:
        pd.DataFrame: DataFrame containing only businesses with "Restaurants" category
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Filter businesses that specifically have "Restaurants" category
        restaurant_mask = df['categories'].fillna('').apply(
            lambda cats: 'Restaurants' in [cat.strip() for cat in cats.split(',')] if cats else False
        )
        
        restaurants_df = df[restaurant_mask].copy()
        
        # Save to CSV
        restaurants_df.to_csv(output_file, index=False)
        
        print(f"\n=== Restaurant Filtering Results ===")
        print(f"Total businesses in input file: {len(df):,}")
        print(f"Businesses with 'Restaurants' category: {len(restaurants_df):,}")
        print(f"Percentage of businesses that are restaurants: {(len(restaurants_df)/len(df)*100):.1f}%")
        print(f"Restaurant data saved to: {output_file}")
        
        # Show breakdown of other categories these restaurants also have
        print(f"\nOther categories found in these restaurants:")
        all_categories = Counter()
        for categories in restaurants_df['categories'].dropna():
            category_list = [cat.strip() for cat in categories.split(',')]
            for cat in category_list:
                if cat != 'Restaurants':  # Exclude the main "Restaurants" category
                    all_categories[cat] += 1
        
        for category, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {category}: {count} businesses")
        
        return restaurants_df
        
    except FileNotFoundError:
        print(f"Error: {csv_file_path} not found. Please run data extraction first.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error filtering restaurants: {e}")
        return pd.DataFrame()


def analyze_categories():
    """
    Analyze and print detailed category statistics from the Santa Barbara businesses CSV file
    """
    try:
        df = pd.read_csv('santa_barbara_businesses.csv')
        category_counts = Counter()
        
        for categories in df['categories'].dropna():
            category_list = [cat.strip() for cat in categories.split(',')]
            category_counts.update(category_list)
        
        print("\n=== Santa Barbara Business Categories Analysis ===")
        print(f"\nTotal number of unique categories: {len(category_counts)}")
        
        print("\nCategories by frequency (showing all):")
        print("-" * 50)
        for category, count in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
            percentage = (count / len(df)) * 100
            print(f"{category}: {count} businesses ({percentage:.1f}%)")
        
        print("\nSummary Statistics:")
        print("-" * 20)
        print(f"Total businesses analyzed: {len(df)}")
        
        categories_per_business = df['categories'].dropna().apply(lambda x: len(x.split(','))).mean()
        print(f"Average categories per business: {categories_per_business:.1f}")
        
        df['category_count'] = df['categories'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
        most_categories = df.nlargest(5, 'category_count')[['name', 'categories', 'category_count']]
        
        print("\nBusinesses with most categories:")
        print("-" * 30)
        for _, row in most_categories.iterrows():
            print(f"{row['name']}: {row['category_count']} categories")
            
    except FileNotFoundError:
        print("Error: santa_barbara_businesses.csv not found. Please run data extraction first.")


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def main():
    """
    Main function to orchestrate the data processing workflow
    """
    # Process businesses
    sb_df = parse_santa_barbara_businesses('yelp_academic_dataset_business.json')
    if not sb_df.empty:
        sb_df = process_business_data(sb_df)
        save_data_to_csv(sb_df, 'santa_barbara_businesses.csv')
        
        # Process reviews
        print("\nFiltering reviews for Santa Barbara businesses...")
        sb_business_ids = set(sb_df['business_id'])
        sb_reviews_df = filter_reviews_for_businesses('yelp_academic_dataset_review.json', sb_business_ids)
        
        if not sb_reviews_df.empty:
            save_data_to_csv(sb_reviews_df, 'santa_barbara_reviews.csv')
            user_ids = process_review_data(sb_reviews_df)
            
            # Process users
            print("\nFiltering user data for reviewers...")
            sb_users_df = filter_users_for_reviews('yelp_academic_dataset_user.json', user_ids)
            if not sb_users_df.empty:
                save_data_to_csv(sb_users_df, 'santa_barbara_users.csv')
                process_user_data(sb_users_df)
            else:
                print("No user data found")
        else:
            print("No reviews found for Santa Barbara businesses")
    else:
        print("\nNo businesses found in Santa Barbara")


def main_restaurants():
    """
    Main function to process restaurants specifically from Santa Barbara data
    """
    print("Starting restaurant data processing...")
    
    # First, get all Santa Barbara businesses
    sb_df = parse_santa_barbara_businesses('yelp_academic_dataset_business.json')
    if not sb_df.empty:
        print(f"Found {len(sb_df):,} businesses in Santa Barbara")
        
        # Filter for restaurants
        restaurants_df = filter_restaurants_from_businesses(sb_df)
        
        if not restaurants_df.empty:
            # Get restaurant business IDs
            restaurant_ids = set(restaurants_df['business_id'])
            
            # Filter reviews for restaurants
            print("\nFiltering reviews for restaurants...")
            restaurant_reviews_df = filter_reviews_for_businesses(
                'yelp_academic_dataset_review.json', 
                restaurant_ids
            )
            
            if not restaurant_reviews_df.empty:
                save_data_to_csv(restaurant_reviews_df, 'santa_barbara_restaurant_reviews.csv')
                
                # Get user IDs from restaurant reviews
                user_ids = set(restaurant_reviews_df['user_id'])
                print(f"Found {len(user_ids):,} unique users who reviewed restaurants")
                
                # Filter users who reviewed restaurants
                print("\nFiltering user data for restaurant reviewers...")
                restaurant_users_df = filter_users_for_reviews(
                    'yelp_academic_dataset_user.json', 
                    user_ids
                )
                
                if not restaurant_users_df.empty:
                    save_data_to_csv(restaurant_users_df, 'santa_barbara_restaurant_users.csv')
                    process_user_data(restaurant_users_df)
                else:
                    print("No user data found for restaurant reviewers")
            else:
                print("No reviews found for restaurants")
        else:
            print("No restaurants found in Santa Barbara data")
    else:
        print("No businesses found in Santa Barbara")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SANTA BARBARA YELP DATA PROCESSING TOOL")
    print("=" * 60)
    
    # Uncomment the following lines to run full data extraction and processing:
    # print("\n" + "="*50)
    # print("EXTRACTING AND PROCESSING ALL DATA")
    # print("="*50)
    # main()
    
    # Uncomment the following lines to run category analysis:
    # print("\n" + "="*50)
    # print("ANALYZING CATEGORIES")
    # print("="*50)
    # analyze_categories()
    
    # Process restaurant data specifically
    print("\n" + "="*50)
    print("PROCESSING RESTAURANT DATA")
    print("="*50)
    main_restaurants()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
