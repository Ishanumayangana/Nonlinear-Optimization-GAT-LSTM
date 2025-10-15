# Advanced AI Pipeline: Nonlinear Optimization → GAT → LSTM

A sophisticated deep learning architecture that combines nonlinear optimization, Graph Attention Networks (GAT), and LSTM with attention mechanisms for advanced sequence prediction tasks.

## 🌟 Overview

This project implements a three-stage AI pipeline that processes optimization problems through graph neural networks and temporal sequence modeling:

1. **Nonlinear Optimization Model** - Generates optimized parameters using differential evolution
2. **Graph Attention Network (GAT)** - Learns path embeddings from graph structures with multi-head attention
3. **LSTM with Attention** - Processes temporal sequences with bidirectional LSTM and attention mechanisms

## 🏗️ Architecture

```
Input Problem → Optimization → Graph Construction → GAT → Embeddings → LSTM → Predictions
```

### Pipeline Stages

#### Stage 1: Nonlinear Optimization
- Solves complex optimization problems using `scipy.optimize`
- Supports multiple optimization paths with differential evolution
- Generates optimized parameter sets for downstream processing

#### Stage 2: Graph Attention Network (GAT)
- Multi-layer GAT with configurable attention heads
- Layer normalization and dropout for regularization
- Learns rich node embeddings from graph structures
- Visualizes graph adjacency matrices and learned representations

#### Stage 3: LSTM with Attention
- Bidirectional LSTM for sequence processing
- Self-attention mechanism for temporal pattern recognition
- Multi-layer feed-forward network for final predictions
- Supports variable sequence lengths

## 📊 Model Specifications

### GAT Architecture
- **Input channels**: 10
- **Hidden channels**: 32
- **Output channels**: 64
- **Attention heads**: 4
- **Number of layers**: 3
- **Total parameters**: ~69K

### LSTM Architecture
- **Input dimension**: 64
- **Hidden dimension**: 128
- **Output dimension**: 64
- **LSTM layers**: 3
- **Bidirectional**: Yes
- **Total parameters**: ~1.1M

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.8+
- CUDA-capable GPU

### Installation


**Install required packages**
```bash
pip install torch torchvision
pip install torch_geometric
pip install scipy scikit-learn matplotlib numpy
```

### Quick Start

1. Open `GAT_LSTM.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. The notebook will:
   - Generate synthetic optimization data
   - Train the GAT model
   - Train the LSTM model
   - Evaluate the complete pipeline
   - Generate visualizations

## 📈 Results

The model achieves excellent performance on synthetic data:

- **Mean Squared Error (MSE)**: 0.000003
- **Mean Absolute Error (MAE)**: 0.001411
- **R² Score**: 0.983

### Visualizations

The notebook generates multiple visualizations:
- Graph adjacency matrix heatmap
- GAT training loss curves
- Path embeddings (PCA projection)
- Temporal sequence patterns
- LSTM training and validation curves
- Prediction vs ground truth scatter plots
- Attention weight heatmaps
- Error distribution histograms

## 🔧 Usage

### Training the Complete Pipeline

```python
# Initialize the complete pipeline
pipeline = CompletePipeline(
    opt_model=opt_model,
    gat_model=gat_model,
    lstm_model=lstm_model
)

# Run on new data
results = pipeline.run_pipeline(
    num_paths=10,
    num_variables=10,
    sequence_length=8
)
```

### Using Individual Components

#### Nonlinear Optimization
```python
opt_model = NonlinearOptimizationModel(num_variables=10)
optimized_params, obj_values = opt_model.optimize_multiple_paths(num_paths=20)
```

#### GAT Training
```python
gat_model = GraphAttentionNetwork(
    in_channels=10,
    hidden_channels=32,
    out_channels=64,
    heads=4,
    num_layers=3
)
training_losses = train_gat(gat_model, graph_data, epochs=100)
```

#### LSTM Training
```python
lstm_model = AttentionLSTM(
    input_dim=64,
    hidden_dim=128,
    output_dim=64,
    num_layers=3,
    bidirectional=True
)
train_losses, val_losses = train_lstm(
    lstm_model, X_train, y_train, X_val, y_val, epochs=150
)
```

## 🎯 Key Features

- ✅ **Modular Design**: Each component can be used independently
- ✅ **Attention Mechanisms**: Both GAT and LSTM use attention for better performance
- ✅ **Comprehensive Visualization**: Multiple plots for analysis and debugging
- ✅ **Flexible Architecture**: Easy to adjust hyperparameters and model structure
- ✅ **Well-Documented**: Detailed comments and markdown explanations
- ✅ **Production-Ready**: Includes training, validation, and evaluation pipelines

## 🔬 Technical Details

### Graph Construction
- Nodes represent optimized parameter sets
- Edges are created based on similarity (top-k nearest neighbors)
- Edge weights represent connection strength

### Training Strategy
- **GAT Training**: 100 epochs with Adam optimizer
- **LSTM Training**: 150 epochs with learning rate scheduling
- **Regularization**: Dropout (0.3) and gradient clipping
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Adam with ReduceLROnPlateau scheduler

### Attention Mechanisms
- **GAT**: Multi-head attention with 4 heads
- **LSTM**: Self-attention over temporal sequences
- Attention weights are visualized for interpretability

## 📊 Performance Metrics

The model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)
- Attention weight analysis
- Error distribution analysis

## 🛠️ Customization

### Modify Model Parameters

```python
# GAT Configuration
gat_model = GraphAttentionNetwork(
    in_channels=10,      # Input feature dimension
    hidden_channels=64,  # Hidden layer size
    out_channels=128,    # Output embedding size
    heads=8,             # Number of attention heads
    num_layers=4         # Number of GAT layers
)

# LSTM Configuration
lstm_model = AttentionLSTM(
    input_dim=128,       # Must match GAT output
    hidden_dim=256,      # LSTM hidden dimension
    output_dim=128,      # Output dimension
    num_layers=4,        # Number of LSTM layers
    bidirectional=True   # Use bidirectional LSTM
)
```

### Adjust Training Parameters

```python
# Training configuration
train_gat(
    model=gat_model,
    data=graph_data,
    epochs=200,          # Number of training epochs
    lr=0.001             # Learning rate
)

train_lstm(
    model=lstm_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=200,
    lr=0.0005
)
```

