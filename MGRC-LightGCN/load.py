import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

# Helper function to create group masks for sensitive attributes
def create_group_mask(users, sensitive_attribute, label_encoder):
    """Create group mask for a sensitive attribute like gender or age."""
    # Apply label encoding for sensitive attribute (e.g., gender, age)
    group_mask = label_encoder.transform(users[sensitive_attribute].values)
    return torch.tensor(group_mask, dtype=torch.long)

# Load dataset for ALiEC, LastFM-1K, MovieLens-1M, MovieLens-100K
def load_data(dataset_name, sensitive_attrs=['gender', 'age']):
    """Load dataset and return edge_index, ratings, group_masks."""
    if dataset_name == 'ALiEC':
        # Load ALiEC dataset (for example, from CSV)
        interactions_df = pd.read_csv('path_to_ALiEC.inter')  # Replace with actual file path
        users_df = pd.read_csv('path_to_ALiEC_users.')   # Users info (gender, age)
        items_df = pd.read_csv('path_to_ALiEC_items.')   # Items info (item data)

        # Create user-item interaction edge list
        edge_index = torch.tensor(interactions_df[['user_id', 'item_id']].values.T, dtype=torch.long)

        # Ratings (Assuming interactions_df has 'rating' column)
        ratings = torch.tensor(interactions_df['rating'].values, dtype=torch.float32)

        # Create group masks for sensitive attributes (gender and age in this case)
        group_masks = {}
        for attr in sensitive_attrs:
            label_encoder = LabelEncoder()
            users_df[attr] = label_encoder.fit_transform(users_df[attr])
            group_masks[attr] = create_group_mask(users_df, attr, label_encoder)

    elif dataset_name == 'lastfm-1k':
        # Load LastFM-1K dataset
        interactions_df = pd.read_csv('path_to_lastfm-1k.inter')  # Replace with actual file path
        users_df = pd.read_csv('path_to_lastfm_users.')      # Users info (gender, age)
        items_df = pd.read_csv('path_to_lastfm_items.')      # Items info (item data)

        # Create user-item interaction edge list
        edge_index = torch.tensor(interactions_df[['user_id', 'item_id']].values.T, dtype=torch.long)

        # Ratings (Assuming interactions_df has 'rating' column)
        ratings = torch.tensor(interactions_df['rating'].values, dtype=torch.float32)

        # Create group masks for sensitive attributes
        group_masks = {}
        for attr in sensitive_attrs:
            label_encoder = LabelEncoder()
            users_df[attr] = label_encoder.fit_transform(users_df[attr])
            group_masks[attr] = create_group_mask(users_df, attr, label_encoder)

    elif dataset_name == 'ml-1m':
        # Load MovieLens 1M dataset
        interactions_df = pd.read_csv('path_to_ml-1m.inter')  # Replace with actual file path
        users_df = pd.read_csv('path_to_ml-1m_users.')  # Users info (gender, age)
        items_df = pd.read_csv('path_to_ml-1m_items.')  # Items info (item data)

        # Create user-item interaction edge list
        edge_index = torch.tensor(interactions_df[['user_id', 'item_id']].values.T, dtype=torch.long)

        # Ratings (Assuming interactions_df has 'rating' column)
        ratings = torch.tensor(interactions_df['rating'].values, dtype=torch.float32)

        # Create group masks for sensitive attributes
        group_masks = {}
        for attr in sensitive_attrs:
            label_encoder = LabelEncoder()
            users_df[attr] = label_encoder.fit_transform(users_df[attr])
            group_masks[attr] = create_group_mask(users_df, attr, label_encoder)

    elif dataset_name == 'ml-100k':
        # Load MovieLens 100K dataset
        interactions_df = pd.read_csv('path_to_ml-100k.inter')  # Replace with actual file path
        users_df = pd.read_csv('path_to_ml-100k_users.')  # Users info (gender, age)
        items_df = pd.read_csv('path_to_ml-100k_items.')  # Items info (item data)

        # Create user-item interaction edge list
        edge_index = torch.tensor(interactions_df[['user_id', 'item_id']].values.T, dtype=torch.long)

        # Ratings (Assuming interactions_df has 'rating' column)
        ratings = torch.tensor(interactions_df['rating'].values, dtype=torch.float32)

        # Create group masks for sensitive attributes
        group_masks = {}
        for attr in sensitive_attrs:
            label_encoder = LabelEncoder()
            users_df[attr] = label_encoder.fit_transform(users_df[attr])
            group_masks[attr] = create_group_mask(users_df, attr, label_encoder)

    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized")

    # Return edge_index, ratings, and group_masks
    return edge_index, ratings, group_masks

# Example usage
if __name__ == "__main__":
    # Load ALiEC dataset
    edge_index, ratings, group_masks = load_data('ALiEC')

    # Print edge_index (user-item interaction edges)
    print("Edge Index: ", edge_index)

    # Print ratings (user-item ratings)
    print("Ratings: ", ratings)

    # Print group masks for sensitive attributes (gender, age)
    print("Group Masks: ", group_masks)
