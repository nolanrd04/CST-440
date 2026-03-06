"""
train_gru_model_pytorch.py - Train a GRU model for keyword spotting using PyTorch.

Handles:
- Loading preprocessed MFCC data (49 frames x 13 coefficients)
- Building and training a GRU(48) classifier for 8 keyword classes
- Evaluating per-class precision/recall/F1 and confusion matrix
- Exporting to ONNX format for deployment
- Generating a C header for Arduino deployment
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
ARDUINO_DIR = os.path.join(os.path.dirname(__file__), "keyword_spotting_arduino")

NUM_CLASSES = 8


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_data():
    """Load preprocessed training, validation, and test data."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)

    # Invert label map: index -> name
    index_to_label = {v: k for k, v in label_map.items()}

    print(f"Training data:   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data:       X={X_test.shape}, y={y_test.shape}")
    print(f"Label map: {label_map}")
    print(f"Number of classes: {len(label_map)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_map, index_to_label


class GRUKeywordSpotter(nn.Module):
    """GRU-based keyword spotting model.

    Architecture: GRU(48) -> Dropout(0.3) -> Dense(num_classes)
    ~9,400 parameters, similar to TensorFlow version.
    """

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super(GRUKeywordSpotter, self).__init__()
        self.hidden_size = hidden_size

        # GRU layer - batch_first=True means input is (batch, seq, features)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # GRU output: (batch, seq_len, hidden_size), hidden: (1, batch, hidden_size)
        output, hidden = self.gru(x)

        # Take the last hidden state (equivalent to return_sequences=False)
        last_hidden = hidden.squeeze(0)  # (batch, hidden_size)

        # Apply dropout and fully connected layer
        out = self.dropout(last_hidden)
        out = self.fc(out)

        return out


def build_model(input_size, num_classes, device):
    """Build and return the GRU model."""
    model = GRUKeywordSpotter(
        input_size=input_size,
        hidden_size=48,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    # Print model summary
    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Create PyTorch DataLoaders for training and validation."""
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0

    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_model(model, train_loader, val_loader, device, epochs=100):
    """Train the model with early stopping and learning rate reduction."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True,
    )

    early_stopping = EarlyStopping(patience=10, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best_weights(model)
            print("Restoring best model weights")
            break

    return history


def evaluate_model(model, X_test, y_test, index_to_label, device):
    """Evaluate the model and print per-class metrics."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        outputs = model(X_test_t)
        test_loss = criterion(outputs, y_test_t).item()
        _, y_pred_classes = torch.max(outputs, 1)
        y_pred_classes = y_pred_classes.cpu().numpy()

    test_acc = (y_pred_classes == y_test).mean()

    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    label_names = [index_to_label[i] for i in range(len(index_to_label))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    # Header
    header = "        " + "  ".join(f"{name[:5]:>5}" for name in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{label_names[i]:>7} " + "  ".join(f"{val:>5}" for val in row)
        print(row_str)

    return test_acc


def export_to_onnx(model, input_shape, device):
    """Export PyTorch model to ONNX format."""
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, input_shape[0], input_shape[1]).to(device)

    onnx_path = os.path.join(os.path.dirname(__file__), "kws_model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    file_size = os.path.getsize(onnx_path)
    print(f"\nSaved ONNX model: {onnx_path} ({file_size} bytes)")
    print(f"ONNX input shape:  (batch, {input_shape[0]}, {input_shape[1]})")
    print(f"ONNX output shape: (batch, {NUM_CLASSES})")

    return onnx_path


def generate_c_header_from_onnx(onnx_path, label_map, index_to_label):
    """Generate C header file for Arduino deployment from ONNX model.

    Includes model bytes, MFCC normalization constants, and label names.
    """
    os.makedirs(ARDUINO_DIR, exist_ok=True)
    header_path = os.path.join(ARDUINO_DIR, "kws_model_data_pytorch.h")

    # Read ONNX model bytes
    with open(onnx_path, 'rb') as f:
        model_bytes = f.read()

    # Load normalization stats
    mean = np.load(os.path.join(DATA_DIR, "mean.npy")).flatten()
    std = np.load(os.path.join(DATA_DIR, "std.npy")).flatten()

    with open(header_path, 'w') as f:
        f.write("// Keyword Spotting ONNX model and configuration\n")
        f.write("// Auto-generated by train_gru_model_pytorch.py - do not edit\n")
        f.write("#ifndef KWS_MODEL_DATA_PYTORCH_H\n")
        f.write("#define KWS_MODEL_DATA_PYTORCH_H\n\n")

        # Model data
        f.write(f"// Model size: {len(model_bytes)} bytes\n")
        f.write("alignas(8) const unsigned char kws_model_onnx[] = {\n")
        for i in range(0, len(model_bytes), 12):
            row = model_bytes[i:i+12]
            hex_vals = ', '.join(f'0x{b:02x}' for b in row)
            if i + 12 < len(model_bytes):
                f.write(f"  {hex_vals},\n")
            else:
                f.write(f"  {hex_vals}\n")
        f.write("};\n\n")
        f.write(f"const unsigned int kws_model_onnx_len = {len(model_bytes)};\n\n")

        # Number of classes
        num_classes = len(label_map)
        f.write(f"const int kNumClasses = {num_classes};\n\n")

        # Label names
        f.write("const char* const kLabelNames[] = {\n")
        for i in range(num_classes):
            name = index_to_label[i]
            comma = "," if i < num_classes - 1 else ""
            f.write(f'  "{name}"{comma}\n')
        f.write("};\n\n")

        # MFCC normalization constants (per-coefficient mean and std)
        f.write(f"const int kNumMfccCoeffs = {len(mean)};\n\n")

        f.write("const float kMfccMean[] = {\n  ")
        f.write(", ".join(f"{v:.6f}f" for v in mean))
        f.write("\n};\n\n")

        f.write("const float kMfccStd[] = {\n  ")
        f.write(", ".join(f"{v:.6f}f" for v in std))
        f.write("\n};\n\n")

        f.write("#endif\n")

    print(f"Saved C header: {header_path}")


def main():
    print("=" * 60)
    print("Keyword Spotting - GRU Model Training (PyTorch)")
    print("=" * 60)

    # Get device
    device = get_device()
    print(f"Training on: {device}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, label_map, index_to_label = load_data()

    # Build model
    input_size = X_train.shape[2]  # 13 (MFCC coefficients)
    model = build_model(input_size, len(label_map), device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64)

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    train_model(model, train_loader, val_loader, device, epochs=100)

    # Evaluate
    test_acc = evaluate_model(model, X_test, y_test, index_to_label, device)

    # Export to ONNX
    print("\n" + "=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)
    input_shape = (X_train.shape[1], X_train.shape[2])  # (49, 13)
    onnx_path = export_to_onnx(model, input_shape, device)

    # Generate C header
    generate_c_header_from_onnx(onnx_path, label_map, index_to_label)

    # Save PyTorch model
    pytorch_path = os.path.join(os.path.dirname(__file__), "kws_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': 48,
        'num_classes': len(label_map),
        'label_map': label_map,
    }, pytorch_path)
    print(f"Saved PyTorch model: {pytorch_path}")

    print("\n" + "=" * 60)
    print(f"DONE - Test accuracy: {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
