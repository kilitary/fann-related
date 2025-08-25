# FANN Neural Network Training Tools

A collection of command-line utilities for training and testing neural networks using the FANN (Fast Artificial Neural Network) library.

## Overview

This project provides various tools for working with artificial neural networks, including:

- **Training algorithms**: RPROP, QuickProp, Batch training, Incremental training
- **Simulated Annealing**: Advanced training with temperature control and auto-tuning
- **Network mutation**: Random network configuration testing
- **Data preprocessing**: Input/output scaling and jittering for better generalization
- **Real-time monitoring**: Live training progress with gnuplot integration

## Features

- **Multiple training algorithms** with automatic parameter tuning
- **Simulated annealing** implementation for escaping local minima
- **Data jittering** to improve network generalization
- **Real-time parameter adjustment** during training
- **Cascaded training** support
- **Hit rate optimization** over MSE minimization
- **Network configuration mutation** for architecture optimization
- **Training data classification** and balancing

## Requirements

- FANN library (Fast Artificial Neural Network)
- GCC/G++ compiler
- Standard C/C++ libraries

## Building

```bash
make all
```

## Utilities

### Core Tools

- **train.exe** - Main training utility with multiple algorithms
- **run.exe** - Run trained networks on test data  
- **create.exe** - Create new network architectures
- **mutate.exe** - Mutate network configurations for optimization
- **data.exe** - Data preprocessing and analysis

### Specialized Tools

- **cascade.exe** - Cascaded training implementation
- **find.exe** - Network architecture search
- **fann_nor.exe** - Normalized training

## Usage Examples

### Basic Training
```bash
./train.exe -r train_data.dat  # RPROP training
./train.exe -q train_data.dat  # QuickProp training
./train.exe -s train_data.dat  # Simulated Annealing
```

### Advanced Options
```bash
./train.exe -j 5 -v 100 train_data.dat  # With 5% jittering, report every 100 epochs
./train.exe -a train_data.dat            # Auto-tuning mode
```

### Running Networks
```bash
./run.exe network.net test_data.dat
```

## Training Data Format

Uses standard FANN training data format:
```
<num_train_data> <num_input> <num_output>
<input_1> <input_2> ... <input_n>
<output_1> <output_2> ... <output_m>
...
```

## Key Features

### Simulated Annealing
- Temperature-based training for escaping local minima
- Automatic cooling/heating strategies
- Real-time temperature adjustment

### Data Jittering
- Adds controlled noise to training data
- Improves network generalization
- Configurable noise levels (typically 5-8%)

### Auto-tuning
- Automatic parameter adjustment based on training progress
- Dynamic algorithm switching (RPROP ↔ QuickProp ↔ SA)
- Learning rate and momentum adaptation

### Hit Rate Optimization
- Focuses on classification accuracy over MSE
- Automatic network saving on hit rate improvements
- Configurable bit-fail limits

## File Formats

- **.net** - FANN network files
- **.dat** - Training/test data files  
- **_hist.dat** - Training history for plotting
- **.plt** - GnuPlot configuration files

## Monitoring and Visualization

The tools generate data compatible with GnuPlot for real-time visualization of:
- Training progress
- MSE evolution
- Hit rate trends
- Parameter changes

## License

Public domain - use freely for any purpose.

## Notes

- Optimized for binary classification tasks
- Includes forex trading specific optimizations
- Cross-platform compatible (Linux/Windows)
- Real-time training control via keyboard input

МОЛЧАТЬ!!"1
