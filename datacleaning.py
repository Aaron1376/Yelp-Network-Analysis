import json
import pandas as pd
from collections import Counter

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

if __name__ == "__main__":
    # Run data extraction and processing
    main()
    
    # Run category analysis
    analyze_categories()
