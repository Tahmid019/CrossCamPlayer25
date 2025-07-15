from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

def match_players(detections_a, detections_b):
    mapping = {}
    features_a, features_b = [], []

    common_frames = sorted(set(detections_a) & set(detections_b))
    for f in common_frames:
        features_a.extend([d['feature'] for d in detections_a[f]])
        features_b.extend([d['feature'] for d in detections_b[f]])

    if not features_a or not features_b:
        return {}

    sim_matrix = cosine_similarity(features_a, features_b)
    cost_matrix = 1 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return {f'tacticam_player_{i}': f'broadcast_player_{j}' for i, j in zip(row_ind, col_ind)}
