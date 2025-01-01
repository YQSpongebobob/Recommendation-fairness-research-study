import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def test_model(model, test_data, edge_index, group_masks=None, device='cpu'):
    """
    Test the MGRC-LightGCN model.

    Args:
        model (nn.Module): The trained MGRC-LightGCN model.
        test_data (dict): A dictionary with 'user_ids', 'item_ids', and 'ratings' for the test set.
        edge_index (torch.Tensor): Edge index for the graph.
        group_masks (dict, optional): Group masks for sensitive attributes.
        device (str): Device to run the test ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    model.eval()
    model.to(device)

    # Extract test data
    user_ids = test_data['user_ids'].to(device)
    item_ids = test_data['item_ids'].to(device)
    true_ratings = test_data['ratings'].to(device)

    # Make predictions
    with torch.no_grad():
        predicted_ratings = model.predict(user_ids, item_ids, edge_index.to(device), group_masks)

    # Convert predictions and true ratings to numpy for evaluation
    predicted_ratings_np = predicted_ratings.cpu().numpy()
    true_ratings_np = true_ratings.cpu().numpy()

    # Compute evaluation metrics
    mse = mean_squared_error(true_ratings_np, predicted_ratings_np)
    rmse = sqrt(mse)
    mae = mean_absolute_error(true_ratings_np, predicted_ratings_np)

    # Fairness evaluation (if group masks are provided)
    fairness_metrics = {}
    if group_masks:
        for attr, mask in group_masks.items():
            group_ids = mask[user_ids].to(device)  # Group IDs for the test users
            unique_groups = torch.unique(group_ids)

            group_errors = []
            for group in unique_groups:
                group_mask = (group_ids == group)
                group_true = true_ratings[group_mask].cpu().numpy()
                group_pred = predicted_ratings[group_mask].cpu().numpy()
                group_mse = mean_squared_error(group_true, group_pred)
                group_errors.append(group_mse)

            fairness_metrics[attr] = group_errors

    # Prepare results
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
    }

    if fairness_metrics:
        results['Fairness'] = fairness_metrics

    return results


# Example usage
if __name__ == "__main__":
    # Load the test dataset
    test_data = {
        'user_ids': torch.tensor([10, 20, 30]),  # Example test user IDs
        'item_ids': torch.tensor([5, 15, 25]),  # Example test item IDs
        'ratings': torch.tensor([4.0, 3.5, 5.0])  # Example true ratings
    }

    # Load the trained model
    model = torch.load('path_to_trained_model.pth')  # Replace with your model path
    model.eval()

    # Example graph edge index and group masks
    edge_index = torch.tensor([
        [0, 1, 2],
        [1, 2, 3]
    ])  # Replace with actual edge index

    group_masks = {
        'gender': torch.randint(0, 2, (100,)),  # Dummy gender groups
        'age': torch.randint(0, 4, (100,))      # Dummy age groups
    }

    # Test the model
    results = test_model(model, test_data, edge_index, group_masks, device='cpu')

    # Print evaluation results
    print("Test Results:")
    print(f"MSE: {results['MSE']}")
    print(f"RMSE: {results['RMSE']}")
    print(f"MAE: {results['MAE']}")

    if 'Fairness' in results:
        print("Fairness Metrics:")
        for attr, group_errors in results['Fairness'].items():
            print(f"  {attr}: {group_errors}")
