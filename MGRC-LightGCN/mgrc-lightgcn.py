import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.linear_model import LogisticRegression


class MGRC_LightGCN_with_Causal_Inference(MGRC_LightGCN):
    """Multi-Group Representation Calibration with LightGCN and Causal Inference."""

    def __init__(self, n_users, n_items, embedding_size, hidden_size, n_layers, sensitive_attrs):
        super(MGRC_LightGCN_with_Causal_Inference, self).__init__(n_users, n_items, embedding_size, hidden_size,
                                                                  n_layers, sensitive_attrs)

        # Add a propensity score model for causal inference
        self.propensity_model = LogisticRegression()

    def fit_propensity_model(self, user_item_features, sensitive_attribute_labels):
        """Fit the propensity score model."""
        self.propensity_model.fit(user_item_features, sensitive_attribute_labels)

    def calculate_propensity_scores(self, user_item_features):
        """Calculate the propensity scores for each instance."""
        return torch.tensor(self.propensity_model.predict_proba(user_item_features)[:, 1])

    def causal_adjustment(self, user_item_embeddings, propensity_scores):
        """Adjust embeddings for causal fairness using propensity scores."""
        adjusted_embeddings = user_item_embeddings / (1 + propensity_scores.view(-1, 1))
        return adjusted_embeddings

    def forward(self, edge_index, group_masks=None):
        """Forward pass for MGRC-LightGCN with causal adjustments."""
        user_item_embeddings = torch.cat([
            self.user_embeddings.weight,
            self.item_embeddings.weight
        ], dim=0)

        # Pre-training step
        user_item_embeddings = self.pretrain_layer(user_item_embeddings)

        # Calculate propensity scores (causal adjustment)
        user_item_features = torch.cat([user_item_embeddings, user_item_embeddings],
                                       dim=0)  # Example feature concatenation
        propensity_scores = self.calculate_propensity_scores(user_item_features)

        # Adjust for causal inference
        adjusted_embeddings = self.causal_adjustment(user_item_embeddings, propensity_scores)

        all_embeddings = [adjusted_embeddings]

        # GCN propagation through layers
        for gcn in self.gcn_layers:
            adjusted_embeddings = gcn(adjusted_embeddings, edge_index)
            adjusted_embeddings = self.central_layer(adjusted_embeddings)
            all_embeddings.append(adjusted_embeddings)

        # Aggregate multi-layer embeddings
        stacked_embeddings = torch.cat(all_embeddings, dim=1)
        final_embeddings = self.aggr_layer(stacked_embeddings)

        # Adjust for sensitive attributes if masks are provided
        if group_masks:
            for attr, mask in group_masks.items():
                group_embedding = self.group_embeddings[attr](mask)
                final_embeddings += self.filter_layer(group_embedding)

        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_embeddings, item_embeddings

    def calculate_loss(self, user_ids, item_ids, ratings, edge_index, group_masks=None):
        """Calculate prediction loss with causal fairness adjustments."""
        user_embeddings, item_embeddings = self.forward(edge_index, group_masks)

        user_embed = user_embeddings[user_ids]
        item_embed = item_embeddings[item_ids]

        pred_ratings = (user_embed * item_embed).sum(dim=1)
        mse_loss = self.mse_loss(pred_ratings, ratings)

        fairness_loss = 0.0
        if group_masks:
            for attr, mask in group_masks.items():
                group_preds = self.dis_layer(self.group_embeddings[attr](mask))
                fairness_loss += self.bce_loss(group_preds, torch.zeros_like(group_preds))

        return mse_loss + fairness_loss

    def predict(self, user_ids, item_ids, edge_index, group_masks=None):
        """Make predictions for user-item pairs with causal adjustments."""
        user_embeddings, item_embeddings = self.forward(edge_index, group_masks)

        user_embed = user_embeddings[user_ids]
        item_embed = item_embeddings[item_ids]

        scores = (user_embed * item_embed).sum(dim=1)
        return scores

    def get_sensitive_attributes(self, user_ids, group_masks):
        """Obtain sensitive attribute embeddings for specified users."""
        sensitive_embeddings = {}
        for attr, mask in group_masks.items():
            sensitive_embeddings[attr] = self.group_embeddings[attr](mask[user_ids])
        return sensitive_embeddings


# Example usage
if __name__ == "__main__":
    n_users = 100
    n_items = 200
    embedding_size = 64
    hidden_size = 32
    n_layers = 3

    # Example sensitive attributes: gender, age (dummy groups)
    sensitive_attrs = {
        'gender': ['male', 'female'],
        'age': ['<18', '18-35', '36-50', '>50']
    }
    group_masks = {
        'gender': torch.randint(0, 2, (n_users + n_items,)),
        'age': torch.randint(0, 4, (n_users + n_items,)),
    }

    model = MGRC_LightGCN_with_Causal_Inference(n_users, n_items, embedding_size, hidden_size, n_layers,
                                                sensitive_attrs)

    # Example graph data
    edge_index = torch.tensor([
        [0, 1, 2],
        [1, 2, 3]
    ])  # Dummy edge index

    user_ids = torch.tensor([0, 1])
    item_ids = torch.tensor([0, 1])
    ratings = torch.tensor([4.0, 5.0])

    # Fit the propensity model to sensitive attributes
    user_item_features = torch.randn((n_users + n_items, embedding_size))
    sensitive_attribute_labels = torch.randint(0, 2, (n_users + n_items,))
    model.fit_propensity_model(user_item_features, sensitive_attribute_labels)

    # Training step
    loss = model.calculate_loss(user_ids, item_ids, ratings, edge_index, group_masks)
    print(f"Loss: {loss.item()}")

    predictions = model.predict(user_ids, item_ids, edge_index, group_masks)
    print(f"Predictions: {predictions}")

    sensitive_embeddings = model.get_sensitive_attributes(user_ids, group_masks)
    print(f"Sensitive Attribute Embeddings: {sensitive_embeddings}")
