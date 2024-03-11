import torch

def compensate_node(tensor, node_idx):
    num_frames, num_nodes, node_dim = tensor.shape
    
    # Extract the coordinates of the specific node across all frames
    node_coords = tensor[:, node_idx, :]
        
    # Find the indices of frames where the node is missing ([0,0,0,0])
    missing_frames = (node_coords == torch.zeros(node_dim).to(node_coords.device)).all(dim=1)
    
    # Loop through each frame to compensate missing coordinates
    for i in range(num_frames):
        if missing_frames[i]:
            # Find the nearest previous frame where the node is not missing
            prev_frame = i - 1
            while prev_frame >= 0 and missing_frames[prev_frame]:
                prev_frame -= 1
            
            # Find the nearest next frame where the node is not missing
            next_frame = i + 1
            while next_frame < num_frames and missing_frames[next_frame]:
                next_frame += 1
            
            # If both previous and next frames are found, interpolate
            if prev_frame >= 0 and next_frame < num_frames:
                alpha = (i - prev_frame) / (next_frame - prev_frame)
                tensor[i, node_idx, :] = (1 - alpha) * node_coords[prev_frame, :] + alpha * node_coords[next_frame, :]
            
            # If only previous frame is found, use its coordinates
            elif prev_frame >= 0:
                tensor[i, node_idx, :] = node_coords[prev_frame, :]
            
            # If only next frame is found, use its coordinates
            elif next_frame < num_frames:
                tensor[i, node_idx, :] = node_coords[next_frame, :]
    return tensor