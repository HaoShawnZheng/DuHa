import torch
import numpy as np

def get_labels_start_end_time_tensor(frame_wise_labels, bg_class): # Assuming bg_class as indices (int) now
    # Initialize lists to collect label information
    labels = torch.empty(0, dtype=torch.long).to(frame_wise_labels.device)  # Empty tensor for labels, assuming labels are long integers
    starts = torch.empty(0, dtype=torch.long).to(frame_wise_labels.device)  # Empty tensor for start indices
    ends = torch.empty(0, dtype=torch.long).to(frame_wise_labels.device)    # Empty tensor for end indices
    last_label = frame_wise_labels[0]
    
    # Handle the start case
    if last_label not in bg_class:
        labels = torch.cat((labels, frame_wise_labels[0:1]))  # Adding the first label if it's not background
        starts = torch.cat((starts, torch.zeros(1, dtype=torch.long).to(frame_wise_labels.device)))  # Start index is 0
    
    for i in range(1, frame_wise_labels.size(0)):  # Start from 1 as we have already checked the 0th element
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i].item() not in bg_class:  # Use .item() to get Python number from tensor
                labels = torch.cat((labels, frame_wise_labels[i:i+1]))
                starts = torch.cat((starts, torch.tensor([i], dtype=torch.long).to(frame_wise_labels.device)))
            if last_label.item() not in bg_class:
                ends = torch.cat((ends, torch.tensor([i], dtype=torch.long).to(frame_wise_labels.device)))
            last_label = frame_wise_labels[i]
    
    # Handle the end case
    if last_label.item() not in bg_class:
        ends = torch.cat((ends, torch.tensor([frame_wise_labels.size(0)], dtype=torch.long).to(frame_wise_labels.device)))  # Note the change here to tensor syntax
    
    return labels, starts, ends

def levenstein_tensor(p, y, norm=False):
    # p and y are tensors containing indices of actions, not strings or other objects
    m_row = p.size(0)
    n_col = y.size(0)
    
    # Initialize the matrix with zeros and the proper type (long for indices)
    D = torch.zeros((m_row+1, n_col+1), dtype=torch.float).to(p.device)
    
    # Populate the matrix with the initial conditions for Levenshtein distance
    for i in range(1, m_row + 1):
        D[i, 0] = i
    for j in range(1, n_col + 1):
        D[0, j] = j
    
    # Compute Levenshtein distance
    for i in range(1, m_row + 1):
        for j in range(1, n_col + 1):
            if p[i - 1] == y[j - 1]:
                cost = 0
            else:
                cost = 1
            D[i, j] = min(D[i - 1, j] + 1,  # Deletion
                          D[i, j - 1] + 1,  # Insertion
                          D[i - 1, j - 1] + cost)  # Substitution
    
    # If normalization is requested, return the normalized score
    if norm:
        score = (1 - D[m_row, n_col] / max(m_row, n_col)) * 100
    else:
        score = D[m_row, n_col]
    
    return score.item()  # Return the value as a Python float


def edit_score_tensor(recognized, ground_truth, norm=True, bg_class=[-1]):
    # Convert recognized and ground_truth to tensor if they are not already
    recognized = torch.as_tensor(recognized)
    ground_truth = torch.as_tensor(ground_truth)

    P, _, _ = get_labels_start_end_time_tensor(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time_tensor(ground_truth, bg_class)
    return levenstein_tensor(P, Y, norm)

def f_score(recognized, ground_truth, overlap, bg_class=[-1]):
    p_label, p_start, p_end = get_labels_start_end_time_tensor(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time_tensor(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = torch.zeros(len(y_label)).to(recognized.device)

    for j in range(len(p_label)):
        # Using torch operations to find the intersection and union
        intersection = torch.min(p_end[j], y_end) - torch.max(p_start[j], y_start).to(recognized.device)
        union = torch.max(p_end[j], y_end) - torch.min(p_start[j], y_start).to(recognized.device)
        
        # Calculating IoU scores for the predicted segment against all ground truth segments
        valid_union_mask = union > 0
        IoU = torch.zeros(len(y_label)).to(recognized.device)  # initialize IoU scores with zeros
        IoU[valid_union_mask] = (intersection[valid_union_mask].float() / union[valid_union_mask].float()) * torch.tensor([p_label[j] == y_label[x] for x in range(len(y_label))]).to(recognized.device)

        # Get the best scoring segment index and IoU value
        max_IoU, idx = torch.max(IoU, dim=0)

        if max_IoU >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(y_label) - hits.sum().item()  # Convert the sum of hits tensor to a Python scalar
    return float(tp), float(fp), float(fn)

def f1_score(recognized, ground_truth, bg_class=[-1]):
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    f1_list = []
    for s in range(len(overlap)):
        tp1, fp1, fn1 = f_score(recognized, ground_truth, overlap[s])
        tp[s] += tp1
        fp[s] += fp1
        fn[s] += fn1
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        f1_list.append(f1)
    return f1_list

def compute_accuracy(pred, gt):
    correct_predictions = (pred == gt.squeeze()).sum().item()
    total_frames = gt.shape[0]
    accuracy = correct_predictions/total_frames
    return accuracy