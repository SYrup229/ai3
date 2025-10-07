from data_loader import CardDataset

# Load and split
dataset = CardDataset('project/data/processed')
dataset.load_and_split()

# Get data for training
X_train, y_train = dataset.get_train_data()
X_val, y_val = dataset.get_val_data()
X_test, y_test = dataset.get_test_data()

# Save splits for reproducibility
dataset.save_splits('project/data/splits')