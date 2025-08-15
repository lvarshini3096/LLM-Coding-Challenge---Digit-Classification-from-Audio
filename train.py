import torch
import torch.nn as nn
from data_preparation import create_dataloaders
from model import CNNClassifier
import time
from sklearn.metrics import classification_report, confusion_matrix

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    num_samples = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        num_samples += labels.size(0)

    avg_loss = total_loss / num_samples
    accuracy = correct_predictions / num_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    num_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            num_samples += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_samples
    accuracy = correct_predictions / num_samples
    return avg_loss, accuracy, all_labels, all_predictions

def main():
    """
    Main function to train and validate the CNN model.
    """
    # --- Configuration ---
    DATA_DIR = "free-spoken-digit-dataset/recordings"
    BATCH_SIZE = 64
    SPLIT_RATIOS = (0.8, 0.1, 0.1)
    LEARNING_RATE = 0.001
    EPOCHS = 10
    NUM_CLASSES = 10
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # --- Data Loading ---
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            split_ratios=SPLIT_RATIOS
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- Model, Loss, Optimizer ---
    model = CNNClassifier(num_classes=NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_state = None

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        
        epoch_duration = time.time() - start_time
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"  -> New best validation accuracy: {val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | Duration: {epoch_duration:.2f}s")

    print("\n--- Training Finished ---")

    # --- Final Evaluation on Test Set ---
    if best_model_state:
        print("\nLoading best model for final test evaluation...")
        model.load_state_dict(best_model_state)

        test_loss, test_acc, all_labels, all_preds = evaluate(model, test_loader, loss_fn, device)
        print(f"\nFinal Test Results:")
        print(f"  - Test Loss: {test_loss:.4f}")
        print(f"  - Test Accuracy: {test_acc:.4f}")

        print("\nClassification Report:")
        target_names = [str(i) for i in range(NUM_CLASSES)]
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        torch.save(best_model_state, "fsdd_cnn_model.pth")
        print("\nBest model saved to fsdd_cnn_model.pth")
    else:
        print("\nNo best model was saved. Skipping test evaluation.")

if __name__ == '__main__':
    main()
