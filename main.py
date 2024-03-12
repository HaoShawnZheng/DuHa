import torch
import torch.nn as nn
import torch.nn.functional as F
import network as network
import os
import torch.optim as optim
import random
import numpy as np
from utils.node_compensate import compensate_node
from utils.eval import edit_score_tensor, f1_score, compute_accuracy
import difflib
import argparse

torch.set_printoptions(threshold=10_000)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def load_sample(root, stage, task_level, view, index, device):
    features = torch.load(root + stage + '_features_' + view + '_' + str(index) + '.pt')
    features = features.to(device)
    i3d_features = torch.load(root + stage + '_i3d_features_' + view + '_' + str(index) + '.pt')
    i3d_features = i3d_features.float().to(device)
    edge_indices = torch.load(root + stage + '_edge_indices_' + view + '_' + str(index) + '.pt')
    edge_indices = edge_indices.to(device)
    lh_labels = torch.load(root + stage + '_lh_' + task_level + '_labels_' + view + '_' + str(index) + '.pt')
    lh_labels = lh_labels.to(device)
    rh_labels = torch.load(root + stage + '_rh_' + task_level + '_labels_' + view + '_' + str(index) + '.pt')
    rh_labels = rh_labels.to(device)
    return features, i3d_features, edge_indices, lh_labels, rh_labels

def get_segments(sequence):
    segmented_sequence, _ = torch.unique_consecutive(sequence, return_counts=True)
    return segmented_sequence

def edit_distance_loss(str1, str2):
    s = difflib.SequenceMatcher(None, str1, str2)
    return 1-s.ratio()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 2000

parser = argparse.ArgumentParser()
parser.add_argument('--element', default='av')
parser.add_argument('--view', default="M0")
parser.add_argument('--data_root', default='./data/data_av/')

args = parser.parse_args()

if not os.path.exists('./data'):
    os.makedirs('./data', exist_ok=True)
    print(f"The directory './data' was created.")
else:
    print(f"The directory './data' already exists.")

if not os.path.exists('./output'):
    os.makedirs('./output', exist_ok=True)
    print(f"The directory './output' was created.")
else:
    print(f"The directory './output' already exists.")

if not os.path.exists('./log'):
    os.makedirs('./log', exist_ok=True)
    print(f"The directory './log' was created.")
else:
    print(f"The directory './log' already exists.")

if __name__ == '__main__':
    log_file = open(f'./log/log_{args.element}_{args.view}.txt','a')

    num_layers = 4
    hidden_dim = 64
    node_initial_dim = 4
    node_feature_dim = 128
    edge_feature_dim = 128
    num_transform_layers = 4

    if args.element == 'av':
        num_classes = 11
    elif args.element == 'mo':
        num_classes = 29
    elif args.element == 'to':
        num_classes = 26
    elif args.element == 'tl':
        num_classes = 5
    else:
        print('The input element is wrong.')

    model = network.BimanualActionPredictionNetwork(node_initial_dim=node_initial_dim, node_dim=node_feature_dim, edge_dim=edge_feature_dim, num_layers=num_layers, hidden_dim=hidden_dim, num_transform_layers=num_transform_layers, num_classes = num_classes)
    model.to(device)
    
    train_data_length = 32
    test_data_length = 6

    criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 200
    best_performance = 0
    for epoch in range(n_epochs):
        model.train()
        
        train_list = list(range(train_data_length))
        random.shuffle(train_list)

        left_hand_accuracies = []
        right_hand_accuracies = []
        left_hand_edit = []
        right_hand_edit = []
        left_hand_f1 = []
        right_hand_f1 = []
        loss_total = []

        for i in range(len(train_list)):
            index = train_list[i]
            
            node_features, i3d_features, edge_indices, lh_label, rh_label = load_sample(args.data_root, 'train', 'aa', args.view, index, device)
            node_features = compensate_node(node_features, 49)
            node_features = compensate_node(node_features, 50)
            
            optimizer.zero_grad()

            hand_0_action_class , hand_1_action_class = model(node_features.unsqueeze(0), edge_indices.unsqueeze(0), i3d_features.unsqueeze(0))
            loss = criterion(hand_0_action_class.view(-1,num_classes), lh_label) + criterion(hand_1_action_class.view(-1,num_classes), rh_label)
            loss += 0.15*torch.mean(torch.clamp(mse_criterion(F.log_softmax(hand_0_action_class[:,1:,:], dim=-1), F.log_softmax(hand_0_action_class[:,:-1,:],dim=-1)), min=0, max=16)) + 0.15*torch.mean(torch.clamp(mse_criterion(F.log_softmax(hand_1_action_class[:,1:,:], dim=-1), F.log_softmax(hand_1_action_class[:,:-1,:], dim=-1)), min=0, max=16))
            _, lh_predicted_labels = torch.max(hand_0_action_class.view(-1,num_classes),dim=1)
            _, rh_predicted_labels = torch.max(hand_1_action_class.view(-1,num_classes),dim=1)
            
            lh_label_segment = get_segments(lh_label)
            rh_label_segment = get_segments(rh_label)
            lh_predicted_segment = get_segments(lh_predicted_labels)
            rh_predicted_segment = get_segments(rh_predicted_labels)
            lh_edit_loss = edit_distance_loss(lh_label_segment, lh_predicted_segment)
            rh_edit_loss = edit_distance_loss(rh_label_segment, rh_predicted_segment)
            loss += torch.tensor(lh_edit_loss)
            loss += torch.tensor(rh_edit_loss)

            lh_accuracy = compute_accuracy(lh_predicted_labels,lh_label)
            rh_accuracy = compute_accuracy(rh_predicted_labels,rh_label)
            lh_edit_score = edit_score_tensor(lh_predicted_labels,lh_label)
            rh_edit_score = edit_score_tensor(rh_predicted_labels,rh_label)
            lh_f1_score = f1_score(lh_predicted_labels,lh_label)
            rh_f1_score = f1_score(rh_predicted_labels,rh_label)
            
            left_hand_accuracies.append(lh_accuracy)
            right_hand_accuracies.append(rh_accuracy)
            left_hand_edit.append(lh_edit_score)
            right_hand_edit.append(rh_edit_score)
            left_hand_f1.append(lh_f1_score)
            right_hand_f1.append(rh_f1_score)
            loss_total.append(loss)

            loss.backward()

            optimizer.step()
        
        left_hand_average_accuracy = sum(left_hand_accuracies)/len(train_list)
        right_hand_average_accuracy = sum(right_hand_accuracies)/len(train_list)
        left_hand_average_edit = sum(left_hand_edit)/len(train_list)
        right_hand_average_edit = sum(right_hand_edit)/len(train_list)
        left_hand_f1 = np.stack(left_hand_f1, axis=0)
        right_hand_f1 = np.stack(right_hand_f1, axis=0)
        left_hand_average_f1 = np.sum(left_hand_f1, axis=0)/len(train_list)
        right_hand_average_f1 = np.sum(right_hand_f1, axis=0)/len(train_list)
        loss_average = sum(loss_total)/len(train_list)
        #print(f"Training Epoch [{epoch+1}/{n_epochs}], Loss: {loss_average.item():.4f}, Left_hand_average_accuracy: {left_hand_average_accuracy:.4f}, Right_hand_average_accuracy: {right_hand_average_accuracy:.4f}, Left_hand_average_edit: {left_hand_average_edit:.4f}, Right_hand_average_edit: {right_hand_average_edit:.4f}, Left_hand_average_f1: {left_hand_average_f1}, Right_hand_average_f1: {right_hand_average_f1}"+ '\n')
        #torch.save(model.state_dict(), f"./checkpoints/checkpoint_{epoch}.pth")
        #log_file.write(f"Training Epoch [{epoch+1}/{n_epochs}], Loss: {loss_average.item():.4f}, Left_hand_average_accuracy: {left_hand_average_accuracy:.4f}, Right_hand_average_accuracy: {right_hand_average_accuracy:.4f}, Left_hand_average_edit: {left_hand_average_edit:.4f}, Right_hand_average_edit: {right_hand_average_edit:.4f}, Left_hand_average_f1: {left_hand_average_f1}, Right_hand_average_f1: {right_hand_average_f1}"+ '\n')

        model.eval()
        
        test_list = list(range(test_data_length))
        
        loss_test_total = []
        left_hand_accuracies_test = []
        right_hand_accuracies_test = []
        left_hand_edit_test = []
        right_hand_edit_test = []
        left_hand_f1_test = []
        right_hand_f1_test = []

        with torch.no_grad():
            top3_lh_predictions = []
            top3_rh_predictions = []
            top3_lh_values = []
            top3_rh_values = []
            for j in range(len(test_list)):
                test_index = test_list[j]
                
                node_features_test, i3d_features_test, edge_indices_test, lh_label_test, rh_label_test = load_sample(args.data_root, 'test', 'aa', args.view, test_index, device)
                # object_embeddings = torch.load('./object_embeddings.pt').to(device)
                node_features_test = compensate_node(node_features_test, 49)
                node_features_test = compensate_node(node_features_test, 50)
                              
                hand_0_action_class , hand_1_action_class = model(node_features_test.unsqueeze(0), edge_indices_test.unsqueeze(0), i3d_features_test.unsqueeze(0))

                loss_test = criterion(hand_0_action_class.view(-1,num_classes), lh_label_test) + criterion(hand_1_action_class.view(-1,num_classes), rh_label_test)
                loss_test += 0.15*torch.mean(torch.clamp(mse_criterion(F.log_softmax(hand_0_action_class[:,1:,:], dim=-1), F.log_softmax(hand_0_action_class[:,:-1,:],dim=-1)), min=0, max=16)) + 0.15*torch.mean(torch.clamp(mse_criterion(F.log_softmax(hand_1_action_class[:,1:,:], dim=-1), F.log_softmax(hand_1_action_class[:,:-1,:], dim=-1)), min=0, max=16))

                _, lh_predicted_labels = torch.max(hand_0_action_class.view(-1,num_classes),dim=1)
                _, rh_predicted_labels = torch.max(hand_1_action_class.view(-1,num_classes),dim=1)

                lh_label_segment = get_segments(lh_label_test)
                rh_label_segment = get_segments(rh_label_test)
                lh_predicted_segment = get_segments(lh_predicted_labels)
                rh_predicted_segment = get_segments(rh_predicted_labels)
                lh_edit_loss = edit_distance_loss(lh_label_segment, lh_predicted_segment)
                rh_edit_loss = edit_distance_loss(rh_label_segment, rh_predicted_segment)
                loss_test += torch.tensor(lh_edit_loss)
                loss_test += torch.tensor(rh_edit_loss)

                lh_accuracy = compute_accuracy(lh_predicted_labels,lh_label_test)
                rh_accuracy = compute_accuracy(rh_predicted_labels,rh_label_test)
                lh_edit_score = edit_score_tensor(lh_predicted_labels,lh_label_test)
                rh_edit_score = edit_score_tensor(rh_predicted_labels,rh_label_test)
                lh_f1_score = f1_score(lh_predicted_labels,lh_label_test)
                rh_f1_score = f1_score(rh_predicted_labels,rh_label_test)

                ####save the top 3 predictions######
                top3_lh_predicted_values, top3_lh_predicted_labels = torch.topk(hand_0_action_class.view(-1, num_classes), 3, dim=1)
                top3_rh_predicted_values, top3_rh_predicted_labels = torch.topk(hand_1_action_class.view(-1, num_classes), 3, dim=1)

                top3_lh_predictions.append(top3_lh_predicted_labels)
                top3_rh_predictions.append(top3_rh_predicted_labels)

                top3_lh_values.append(top3_lh_predicted_values)
                top3_rh_values.append(top3_rh_predicted_values)
                #log_file.write('Testing top3_lh_predicted_labels = '+str(top3_lh_predicted_labels) + '\n')
                #log_file.write('Testing top3_rh_predicted_labels = '+str(top3_rh_predicted_labels) + '\n')

                left_hand_accuracies_test.append(lh_accuracy)
                right_hand_accuracies_test.append(rh_accuracy)
                left_hand_edit_test.append(lh_edit_score)
                right_hand_edit_test.append(rh_edit_score)
                left_hand_f1_test.append(lh_f1_score)
                right_hand_f1_test.append(rh_f1_score)
                loss_test_total.append(loss_test)

                #print(f"Testing Epoch [{epoch+1}/{n_epochs}], Step [{j+1}], Loss: {loss_test.item():.4f}, Left_hand_accuracy: {lh_accuracy:.4f}, Right_hand_accuracy: {rh_accuracy:.4f}, Left_hand_edit: {lh_edit_score:.4f}, Right_hand_edit: {rh_edit_score:.4f}, Left_hand_f1: {lh_f1_score}, Right_hand_f1: {rh_f1_score}" + '\n')
        
        loss_average_test = sum(loss_test_total)/len(test_list)
        left_hand_average_accuracy_test = sum(left_hand_accuracies_test)/len(test_list)
        right_hand_average_accuracy_test = sum(right_hand_accuracies_test)/len(test_list)
        left_hand_average_edit_test = sum(left_hand_edit_test)/len(test_list)
        right_hand_average_edit_test = sum(right_hand_edit_test)/len(test_list)
        left_hand_f1_test = np.stack(left_hand_f1_test, axis=0)
        right_hand_f1_test = np.stack(right_hand_f1_test, axis=0)
        left_hand_average_f1_test = np.sum(left_hand_f1_test, axis=0)/len(test_list)
        right_hand_average_f1_test = np.sum(right_hand_f1_test, axis=0)/len(test_list)
        print(f"Testing Epoch [{epoch+1}/{n_epochs}], Loss_average: {loss_average_test.item():.4f}, Left_hand_average_accuracy: {left_hand_average_accuracy_test:.4f}, Right_hand_average_accuracy: {right_hand_average_accuracy_test:.4f}, Left_hand_average_edit: {left_hand_average_edit_test:.4f}, Right_hand_average_edit: {right_hand_average_edit_test:.4f}, Left_hand_average_f1: {left_hand_average_f1_test}, Right_hand_average_f1: {right_hand_average_f1_test}" + '\n')
        log_file.write(f"Testing Epoch [{epoch+1}/{n_epochs}], Loss_average: {loss_average_test.item():.4f}, Left_hand_average_accuracy: {left_hand_average_accuracy_test:.4f}, Right_hand_average_accuracy: {right_hand_average_accuracy_test:.4f}, Left_hand_average_edit: {left_hand_average_edit_test:.4f}, Right_hand_average_edit: {right_hand_average_edit_test:.4f}, Left_hand_average_f1: {left_hand_average_f1_test}, Right_hand_average_f1: {right_hand_average_f1_test}" + '\n')

        average_performance = (left_hand_average_accuracy_test*100 + right_hand_average_accuracy_test*100 + left_hand_average_edit_test + right_hand_average_edit_test)/4
        if average_performance > best_performance:
            best_performance = average_performance           
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': average_performance,
                # include any other state info that you need
            }
            torch.save(checkpoint, f'./output/{args.element}_{args.view}_checkpoint.pth')
            torch.save(top3_lh_predictions, f'./output/{args.element}_{args.view}_lh_predictions.pt')
            torch.save(top3_rh_predictions, f'./output/{args.element}_{args.view}_rh_predictions.pt')
            torch.save(top3_lh_values, f'./output/{args.element}_{args.view}_lh_values.pt')
            torch.save(top3_rh_values, f'./output/{args.element}_{args.view}_rh_values.pt')
    log_file.close()
