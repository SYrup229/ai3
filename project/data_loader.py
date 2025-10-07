"""
Data Loader for Playing Card Rank Classification
Handles: Loading preprocessed images, train/val/test split, data organization
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import pickle


class CardDataset:
    """Dataset loader for preprocessed card images"""
    
    def __init__(self, data_dir, split_ratios=(0.7, 0.15, 0.15), random_state=42):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing preprocessed images (organized by rank folders)
            split_ratios: (train, val, test) ratios, must sum to 1.0
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split_ratios = split_ratios
        self.random_state = random_state
        
        # Validate split ratios
        assert abs(sum(split_ratios) - 1.0) < 0.001, "Split ratios must sum to 1.0"
        
        # Rank mapping
        self.rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.rank_to_label = {rank: idx for idx, rank in enumerate(self.rank_names)}
        self.label_to_rank = {idx: rank for rank, idx in self.rank_to_label.items()}
        
        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
    def load_and_split(self):
        """
        Load all preprocessed images and split into train/val/test
        """
        print("Loading preprocessed images...")
        
        all_images = []
        all_labels = []
        all_files = []
        
        # Load images from each rank folder
        for rank in self.rank_names:
            rank_folder = self.data_dir / rank
            
            if not rank_folder.exists():
                print(f"Warning: Folder for rank '{rank}' not found, skipping...")
                continue
            
            image_files = list(rank_folder.glob("*.png"))
            label = self.rank_to_label[rank]
            
            for img_file in image_files:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    all_images.append(img)
                    all_labels.append(label)
                    all_files.append(str(img_file))
            
            print(f"  Loaded {len(image_files)} images for rank '{rank}'")
        
        print(f"\nTotal images loaded: {len(all_images)}")
        
        # Convert to numpy arrays
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        all_files = np.array(all_files)
        
        # Print class distribution
        print("\nClass distribution:")
        for rank in self.rank_names:
            label = self.rank_to_label[rank]
            count = np.sum(all_labels == label)
            print(f"  {rank}: {count} images")
        
        # First split: train+val vs test
        train_val_ratio = self.split_ratios[0] + self.split_ratios[1]
        test_ratio = self.split_ratios[2]
        
        X_trainval, X_test, y_trainval, y_test, files_trainval, files_test = train_test_split(
            all_images, all_labels, all_files,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=all_labels  # Ensure balanced split across ranks
        )
        
        # Second split: train vs val
        val_ratio_adjusted = self.split_ratios[1] / train_val_ratio
        
        X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
            X_trainval, y_trainval, files_trainval,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=y_trainval
        )
        
        # Store splits
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        self.train_files = files_train.tolist()
        self.val_files = files_val.tolist()
        self.test_files = files_test.tolist()
        
        # Print split statistics
        print("\n" + "="*50)
        print("Dataset split complete:")
        print(f"  Train: {len(X_train)} images ({len(X_train)/len(all_images)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} images ({len(X_val)/len(all_images)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} images ({len(X_test)/len(all_images)*100:.1f}%)")
        
        # Print per-class split
        print("\nPer-rank distribution:")
        print(f"{'Rank':<6} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
        print("-" * 40)
        for rank in self.rank_names:
            label = self.rank_to_label[rank]
            train_count = np.sum(y_train == label)
            val_count = np.sum(y_val == label)
            test_count = np.sum(y_test == label)
            total_count = train_count + val_count + test_count
            print(f"{rank:<6} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
        
        return self
    
    def get_train_data(self):
        """Get training data"""
        return self.X_train, self.y_train
    
    def get_val_data(self):
        """Get validation data"""
        return self.X_val, self.y_val
    
    def get_test_data(self):
        """Get test data"""
        return self.X_test, self.y_test
    
    def save_splits(self, output_dir):
        """
        Save the train/val/test splits to disk for reproducibility
        
        Args:
            output_dir: Directory to save split information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file lists (for tracking which images are in which split)
        splits = {
            'train_files': self.train_files,
            'val_files': self.val_files,
            'test_files': self.test_files,
            'rank_to_label': self.rank_to_label,
            'label_to_rank': self.label_to_rank
        }
        
        with open(output_dir / 'splits.json', 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"\nSplit information saved to {output_dir / 'splits.json'}")
    
    def load_splits(self, splits_file):
        """
        Load previously saved splits
        
        Args:
            splits_file: Path to splits.json file
        """
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        self.train_files = splits['train_files']
        self.val_files = splits['val_files']
        self.test_files = splits['test_files']
        
        # Load images based on file lists
        print("Loading images from saved splits...")
        
        self.X_train, self.y_train = self._load_images_from_files(self.train_files)
        self.X_val, self.y_val = self._load_images_from_files(self.val_files)
        self.X_test, self.y_test = self._load_images_from_files(self.test_files)
        
        print(f"Loaded: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test")
        
        return self
    
    def _load_images_from_files(self, file_list):
        """Helper to load images from a list of file paths"""
        images = []
        labels = []
        
        for file_path in file_list:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                # Extract rank from path
                rank = Path(file_path).parent.name
                labels.append(self.rank_to_label[rank])
        
        return np.array(images), np.array(labels)
    
    def get_sample_images(self, n_samples=5):
        """
        Get sample images from each rank for visualization
        
        Args:
            n_samples: Number of samples per rank
            
        Returns:
            Dictionary mapping rank -> list of images
        """
        samples = {}
        
        for rank in self.rank_names:
            label = self.rank_to_label[rank]
            # Get indices of this rank in training set
            indices = np.where(self.y_train == label)[0]
            
            # Sample random images
            if len(indices) > 0:
                sample_indices = np.random.choice(indices, 
                                                  min(n_samples, len(indices)), 
                                                  replace=False)
                samples[rank] = self.X_train[sample_indices]
        
        return samples
    
    def print_summary(self):
        """Print dataset summary statistics"""
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Number of classes: {len(self.rank_names)}")
        print(f"Class names: {', '.join(self.rank_names)}")
        print(f"Image shape: {self.X_train[0].shape}")
        print(f"\nDataset sizes:")
        print(f"  Training:   {len(self.X_train):>5} images")
        print(f"  Validation: {len(self.X_val):>5} images")
        print(f"  Test:       {len(self.X_test):>5} images")
        print(f"  Total:      {len(self.X_train) + len(self.X_val) + len(self.X_test):>5} images")


def visualize_samples(dataset, n_samples=3):
    """
    Visualize sample images from each rank
    
    Args:
        dataset: CardDataset instance
        n_samples: Number of samples to show per rank
    """
    import matplotlib.pyplot as plt
    
    samples = dataset.get_sample_images(n_samples)
    
    n_ranks = len(dataset.rank_names)
    fig, axes = plt.subplots(n_ranks, n_samples, figsize=(n_samples*2, n_ranks*2))
    
    for i, rank in enumerate(dataset.rank_names):
        if rank in samples:
            for j, img in enumerate(samples[rank]):
                ax = axes[i, j] if n_samples > 1 else axes[i]
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                if j == 0:
                    ax.set_title(f"Rank: {rank}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print("\nSample images saved to 'sample_images.png'")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Load and split dataset
    dataset = CardDataset('data/processed')
    dataset.load_and_split()
    
    # Print summary
    dataset.print_summary()
    
    # Save splits for reproducibility
    dataset.save_splits('data/splits')
    
    # Visualize samples
    visualize_samples(dataset, n_samples=3)
    
    # Access data
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()
    
    print("\nData ready for feature extraction and model training!")