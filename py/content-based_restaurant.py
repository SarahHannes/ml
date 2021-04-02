# Import libraries --------------------------
import pandas as pd
import numpy as np

# Load data --------------------------

# Get userID (PK)
user = pd.read_csv("../input/restaurant-data-with-consumer-ratings/userprofile.csv")
user = user[["userID"]].drop_duplicates()

# Get user_rating, delete unnecessary columns
user_rating = pd.read_csv("../input/restaurant-data-with-consumer-ratings/rating_final.csv")
user_rating = user_rating.drop(['rating', 'service_rating'], axis=1)

# Get placeID (restaurantID) with cuisine info
cuisine = pd.read_csv("../input/restaurant-data-with-consumer-ratings/chefmozcuisine.csv")
# Gather all cuisine types into placeID and spread the cuisine type for each placeID 
# as of now, all has NaN value, next step will insert 1 if the restaurant serve the particular cuisine type or 0 if it does not serve that cuisine
cuisine_wide = cuisine.pivot(index='placeID', columns='Rcuisine', values='Rcuisine')

# filling 1 or 0s for each different cuisine types for every row(restaurant)
for index, row in cuisine.iterrows():
    cuisine_wide.at[row['placeID'], row['Rcuisine']] = 1
cuisine_wide = cuisine_wide.fillna(0)

# Functions --------------------------

def get_user_rating_df(userid):
    """
    input: userID, string
    returns a filtered df of restaurant's placeID and food_rating based on userid
    """
    return user_rating[user_rating.userID==userid]

def get_rated_place_list(userid):
    """
    input: userID, string
    returns a list of placeID which the user had rated
    """
    restaurant_list = []
    filtered_user = get_user_rating_df(userid)
    for restaurant in filtered_user["placeID"]:
        restaurant_list.append(restaurant)
    return restaurant_list

def user_ratedplace_wide(userid):
    """
    input: userID, string
    returns a df of restaurantID which the user had rated (index) and
    the restaurant's cuisine type on for the rest of the columns,
    the value in all columns (except index) are either 1-if the restaurant serve the particular cuisine;
    or 0-if it doesn't serve the cuisine
    """
    restaurant_list = get_rated_place_list(userid)
    # filtering out cuisine_wide according to the restaurant_list
    return cuisine_wide[cuisine_wide.index.isin(restaurant_list)]

def weighted_user_rating(userid):
    """
    input: userID, string
    returns a filtered df containing only the placeID (index col) the user had rated
    and the placeID's cuisine type in wide format.
    The value of all columns are multiplied with the food_rating value from user_rating
    to get the weighted rating
    """
    # create new empty df to store user weighted rating
    weighted_df = pd.DataFrame()
    # filter user_rating(for the userID) -> only retain 2 columns (placeID and food_rating) -> set the index to placeID
    food_rating = get_user_rating_df(userid)[['placeID', "food_rating"]].set_index('placeID')
    # get user_ratedplace_wide (index is placeID)
    past_ratedplace = user_ratedplace_wide(userid)
    # for each of the placeID in user_rating: (index is placeID, row is all the columns in each row)
    for place in food_rating.index:
        if place in past_ratedplace.index:
            #food_rating.at[place, 'food_rating'] is the rating for each restaurant
            rating = past_ratedplace.loc[place] * food_rating.at[place, 'food_rating']
            # append the rating to the new df
            weighted_df = weighted_df.append(rating)
    return weighted_df

def aggregate_rating_by_cuisine(userid):
    """
    input: userID, string
    aggregate all the weighted user rating by cuisine types
    returns df consist of only past rated cuisine type (index) and subtotal column
    """
    # get weighted_user_rating df and transpose it
    transposed_weighted_df = weighted_user_rating(userid).transpose()
    # make non-zero bool masks
    nonzero_mask_series = (transposed_weighted_df != 0).any(axis=1)
    # apply the non-zero bool masks to the weighted_user_rating
    rating_by_cuisine = transposed_weighted_df.loc[nonzero_mask_series]
    # make another column to sum(all rating for each row)
    rating_by_cuisine.insert(0, 'subtotal', rating_by_cuisine.sum(axis=1))
    # return only the total column for all rows - in df form => notice the double []
    return rating_by_cuisine[['subtotal']]

def normalized_rating_by_cuisine(userid):
    """
    input: userID, string
    returns a df containing normalized aggregated rating weight for all cuisine type the user had previously rated
    """
    total_rating = 0
    # get the aggregated_rating_by_cuisine
    user_profile = aggregate_rating_by_cuisine(userid)
    # add col total
    for subtotal in user_profile['subtotal']:
        total_rating += subtotal
    # for each row, we take total/col total (will get percentage)
    user_profile.insert(1, 'normalized', user_profile['subtotal']/total_rating)
    return user_profile[["normalized"]]

def user_notratedplace_wide(userid):
    """
    input: userID, string
    returns a df containing all of the placeID the user had not rated before
    """
    restaurant_list = get_rated_place_list(userid)
    # filtering out to get a df containing only placeID not in restaurant_list
    return cuisine_wide[~cuisine_wide.index.isin(restaurant_list)]

def get_recommendation(userid, topn=5):
    """
    input: userID, string; topn, int
    returns a df containing topn recommended placeID sorted by desc weighted score
    """
    # filter for placeID not in get_rated_place_list
    # index in recom_cand is placeID
    notrated = user_notratedplace_wide(userid)
    recom = notrated.copy()
    # multiply the filtered df with normalized_rating_by_cuisine
    norm_rating = normalized_rating_by_cuisine(userid)
    # index in norm_rating is cuisine type. col[0] is the normalized value from norm_rating
    for cuisine_type, col in norm_rating.iterrows():
        for placeid, row in notrated.iterrows():
            recom.at[placeid, cuisine_type] = recom.at[placeid, cuisine_type]*col[0]
    # drop all restaurants which has 0 in all cols
    nonzero_mask_series = (recom != 0).any(axis=1)
    # apply the non-zero bool masks to the recommendation df
    recom_nonzero = recom.loc[nonzero_mask_series]
    # add new column to sum(multiplied value) by rows
    recom_nonzero.insert(0, 'total_by_place', recom_nonzero.sum(axis=1))
    # sort by sum(multiplied value) desc -> return the sorted by topn .head(topn)
    return recom_nonzero.sort_values(by='total_by_place', ascending=False)[['total_by_place']].head(topn)


# get restaurant recommendation for user
get_recommendation("U1005")