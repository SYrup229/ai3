import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_templates(template_path):
    """
    Load the 14 grayscale template images for rank recognition.
    
    Args:
        template_path: Path to directory containing template images
        
    Returns:
        templates: Dictionary mapping rank name to template image
        template_names: List of template names in order
    """
    templates = {}
    template_dir = Path(template_path)
    
    # Expected template files (adjust names to match your actual files)
    rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'ND']
    
    print("Loading templates...")
    for rank in rank_names:
        # Try different possible filenames
        possible_files = [
            template_dir / f"{rank}.jpg",
            template_dir / f"{rank}.png",
            template_dir / f"{rank.lower()}.jpg",
            template_dir / f"{rank.lower()}.png"
        ]
        
        for template_file in possible_files:
            if template_file.exists():
                template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    templates[rank] = template_img
                    print(f"  ✓ Loaded template: {rank} ({template_img.shape})")
                    break
        
        if rank not in templates:
            print(f"  ✗ WARNING: Template not found for rank: {rank}")
    
    template_names = list(templates.keys())
    print(f"\nTotal templates loaded: {len(templates)}/14")
    
    return templates, template_names


def detect_card_region(img):
    """
    Detect the card region in the image by finding the black card on red background.
    
    Args:
        img: Input image (BGR)
        
    Returns:
        x, y, w, h: Bounding box of the card (or None if not found)
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for black cards (low value/brightness)
    # Black cards have low V (value) regardless of H (hue) or S (saturation)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])  # V < 100 for black cards
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (should be the card)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Verify it's a reasonable size (not tiny noise)
    img_area = img.shape[0] * img.shape[1]
    card_area = w * h
    
    if card_area < img_area * 0.05:  # Card should be at least 5% of image
        return None
    
    return x, y, w, h


def extract_template_features(img, templates, template_names):
    """
    Extract template matching correlation scores as features.
    First detects the card, then crops its top-left corner.
    
    Args:
        img: Input card image (BGR)
        templates: Dictionary of template images
        template_names: List of template names
        
    Returns:
        correlation_scores: Array of 14 correlation scores
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect card region first
    card_bbox = detect_card_region(img)
    
    if card_bbox is None:
        # If card not detected, assume it's EMPTY or use full image
        # Return low scores for all templates
        return np.zeros(len(template_names))
    
    x, y, w, h = card_bbox
    
    # Crop the card region
    card_gray = gray[y:y+h, x:x+w]
    
    # Crop top-left corner of the CARD (not the image)
    # Rank symbol is typically in top 30% height, left 25% width of the card
    card_h, card_w = card_gray.shape
    crop_h = int(card_h * 0.3)
    crop_w = int(card_w * 0.25)
    corner = card_gray[0:crop_h, 0:crop_w]
    
    # Resize corner to match template size (70x125)
    corner_resized = cv2.resize(corner, (70, 125))
    
    # Calculate correlation with each template
    correlation_scores = []
    
    for rank in template_names:
        if rank in templates:
            template = templates[rank]
            
            # Use normalized cross-correlation
            result = cv2.matchTemplate(corner_resized, template, cv2.TM_CCOEFF_NORMED)
            
            # Get the maximum correlation value
            max_corr = np.max(result)
            correlation_scores.append(max_corr)
        else:
            correlation_scores.append(0.0)  # Missing template
    
    return np.array(correlation_scores)


def extract_global_features(img):
    """
    Extract 8 global image features.
    
    Args:
        img: OpenCV image (BGR format)
        
    Returns:
        features: 1D numpy array of 8 global features
    """
    # Resize image to standard size for consistency
    img_resized = cv2.resize(img, (200, 300))
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Feature 1: Mean Red Channel
    mean_red = np.mean(img_resized[:, :, 2])  # BGR format, R is index 2
    
    # Feature 2: Mean Green Channel
    mean_green = np.mean(img_resized[:, :, 1])
    
    # Feature 3: Mean Blue Channel
    mean_blue = np.mean(img_resized[:, :, 0])
    
    # Feature 4: HSV Saturation Mean
    mean_saturation = np.mean(hsv[:, :, 1])
    
    # Feature 5: HSV Value (Brightness) Mean
    mean_value = np.mean(hsv[:, :, 2])
    
    # Feature 6: Edge Density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Feature 7: Color Variance (across all channels)
    color_variance = np.var(img_resized)
    
    # Feature 8: Texture Complexity (using standard deviation of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_complexity = np.std(laplacian)
    
    # Combine all features
    features = np.array([
        mean_red, mean_green, mean_blue,
        mean_saturation, mean_value,
        edge_density, color_variance, texture_complexity
    ])
    
    return features


def load_card_data(data_path, template_path):
    """
    Load card images and extract features (template matching + global features).
    
    Args:
        data_path: Path to directory containing card rank folders
        template_path: Path to directory containing template images
        
    Returns:
        data: Feature array (n_samples, 22 features)
        target: Target labels array
        feature_names: List of feature names
        unique_targets: List of unique card ranks
    """
    # Load templates first
    templates, template_names = load_templates(template_path)
    
    if len(templates) == 0:
        raise ValueError("No templates loaded! Check template path and filenames.")
    
    data_list = []
    target_list = []
    
    # Get all subdirectories (card rank folders)
    data_dir = Path(data_path)
    card_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    
    print(f"\nFound {len(card_folders)} card rank folders:")
    for folder in card_folders:
        print(f"  - {folder.name}")
    
    # Load images from each folder
    for card_folder in card_folders:
        card_rank = card_folder.name
        image_files = list(card_folder.glob('*.jpg')) + list(card_folder.glob('*.png')) + list(card_folder.glob('*.jpeg'))
        
        print(f"\nLoading {len(image_files)} images for rank: {card_rank}")
        
        for img_path in image_files:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: Could not read {img_path}")
                continue
            
            # Extract template matching features (14 features)
            template_features = extract_template_features(img, templates, template_names)
            
            # Extract global features (8 features)
            global_features = extract_global_features(img)
            
            # Combine features (total: 22 features)
            combined_features = np.concatenate([template_features, global_features])
            
            data_list.append(combined_features)
            target_list.append(card_rank)
    
    # Convert to numpy arrays
    data = np.array(data_list)
    target = np.array(target_list)
    
    # Create feature names
    feature_names = []
    
    # Template matching feature names
    for rank in template_names:
        feature_names.append(f'template_match_{rank}')
    
    # Global feature names
    feature_names.extend([
        'mean_red', 'mean_green', 'mean_blue',
        'mean_saturation', 'mean_value',
        'edge_density', 'color_variance', 'texture_complexity'
    ])
    
    unique_targets = sorted(np.unique(target))
    
    print(f"\n{'='*50}")
    print(f"DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples loaded: {len(data)}")
    print(f"Feature vector size: {data.shape[1]}")
    print(f"Number of classes: {len(unique_targets)}")
    print(f"Classes: {unique_targets}")
    print(f"{'='*50}\n")
    
    class CardData:
        pass
    
    cards = CardData()
    cards.data = data
    cards.target = target
    cards.feature_names = feature_names
    cards.unique_targets = unique_targets
    
    return cards


if __name__ == "__main__":
    """Card rank detection - feature exploration with template matching"""
    
    # UPDATE THESE PATHS
    data_path = 'data'  # Folder containing A, 2, 3, ..., K, ND, EMPTY subfolders
    template_path = 'templates'  # Folder containing template images (A.jpg, 2.jpg, etc.)
    
    # Load the card data with features
    print("="*60)
    print("LOADING CARD DATA AND EXTRACTING FEATURES")
    print("="*60)
    cards = load_card_data(data_path, template_path)
    
    # Encode the categorical labels
    le = LabelEncoder()
    coded_labels = le.fit_transform(cards.target)
    
    # Partition the data into training and testing splits
    (trainX, testX, trainY, testY) = train_test_split(
        cards.data, coded_labels,
        test_size=0.25, 
        stratify=cards.target,
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)
    
    print(f"\nTraining set size: {len(trainX)}")
    print(f"Testing set size: {len(testX)}")
    print(f"\nStarting visualization...\n")
    
    # === VISUALIZATIONS ===
    
    # 1. Target distribution
    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x=trainY, color="skyblue")
    ax.set_xticklabels(cards.unique_targets, rotation=45)
    ax.set_xlabel('Card Rank', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Card Ranks in Training Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 2. Template matching scores heatmap (first 14 features)
    plt.figure(figsize=(14, 10))
    template_features = trainX[:, :14]  # First 14 features are template scores
    
    # Create a sample (max 500 samples for readability)
    sample_size = min(500, len(trainX))
    sample_indices = np.random.choice(len(trainX), sample_size, replace=False)
    
    template_sample = template_features[sample_indices]
    labels_sample = trainY[sample_indices]
    
    # Sort by label for better visualization
    sort_idx = np.argsort(labels_sample)
    template_sorted = template_sample[sort_idx]
    
    sns.heatmap(template_sorted.T, cmap='RdYlGn', center=0,
                yticklabels=cards.feature_names[:14],
                xticklabels=False, cbar_kws={'label': 'Normalized Correlation'})
    plt.title('Template Matching Scores Heatmap (Sample of Training Data)', fontsize=14, fontweight='bold')
    plt.ylabel('Template Rank', fontsize=12)
    plt.xlabel('Training Samples (sorted by class)', fontsize=12)
    plt.tight_layout()
    
    # 3. Global features distribution (last 8 features)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    colors = ['skyblue', 'olive', 'gold', 'teal', 'coral', 'purple', 'orange', 'pink']
    
    for i in range(8):
        row, col = i // 4, i % 4
        feature_idx = 14 + i  # Global features start at index 14
        
        sns.histplot(trainX[:, feature_idx], color=colors[i], bins=30, ax=axes[row, col], kde=True)
        axes[row, col].set_xlabel(cards.feature_names[feature_idx], fontsize=10)
        axes[row, col].set_ylabel('Frequency', fontsize=10)
    
    fig.suptitle('Distribution of Global Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 4. Scatter plot: Best template match vs Edge density
    fig = plt.figure(figsize=(12, 8))
    best_template_idx = np.argmax(trainX[:, :14], axis=1)  # Index of best matching template
    edge_density_idx = 14 + 5  # edge_density is the 6th global feature
    
    scatter = plt.scatter(best_template_idx, trainX[:, edge_density_idx], 
                         c=trainY, cmap='tab20', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Card Rank (encoded)', ticks=range(len(cards.unique_targets)))
    plt.xlabel('Best Matching Template Index', fontsize=12)
    plt.ylabel('Edge Density (normalized)', fontsize=12)
    plt.title('Best Template Match vs Edge Density', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 5. Boxplot: Template matching scores by true rank
    plt.figure(figsize=(14, 6))
    
    # For each class, show the correlation score with its own template
    own_template_scores = []
    class_labels = []
    
    for class_idx, class_name in enumerate(cards.unique_targets):
        if class_name == 'EMPTY':
            # For EMPTY, use max template score (should be low)
            class_mask = (trainY == class_idx)
            max_scores = np.max(trainX[class_mask, :14], axis=1)
            own_template_scores.extend(max_scores)
            class_labels.extend([class_name] * len(max_scores))
        elif class_name in cards.feature_names[:14]:
            # Find the template index for this class
            template_feature_name = f'template_match_{class_name}'
            if template_feature_name in cards.feature_names:
                template_idx = cards.feature_names.index(template_feature_name)
                class_mask = (trainY == class_idx)
                scores = trainX[class_mask, template_idx]
                own_template_scores.extend(scores)
                class_labels.extend([class_name] * len(scores))
    
    if own_template_scores:
        ax = sns.boxplot(x=class_labels, y=own_template_scores, palette='Set2')
        ax.set_xlabel('Card Rank', fontsize=12)
        ax.set_ylabel('Template Match Score (normalized)', fontsize=12)
        ax.set_title('Template Matching Performance: Own Template Score by Rank', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
    
    # 6. Feature correlation heatmap
    plt.figure(figsize=(16, 14))
    
    # Select subset of features for readability
    # All 14 template features + 8 global features
    corr = np.corrcoef(trainX, rowvar=False)
    
    mask = np.triu(np.ones_like(corr), k=1)  # Mask upper triangle
    
    sns.heatmap(corr, annot=False, fmt='.2f',
                xticklabels=cards.feature_names, 
                yticklabels=cards.feature_names,
                cmap='coolwarm', center=0, 
                mask=mask,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    # 7. Feature importance visualization (variance-based)
    plt.figure(figsize=(14, 6))
    feature_vars = np.var(trainX, axis=0)
    
    colors_importance = ['green']*14 + ['blue']*8
    bars = plt.bar(range(len(feature_vars)), feature_vars, color=colors_importance, alpha=0.7)
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.title('Feature Variance (Higher = More Discriminative)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(cards.feature_names)), cards.feature_names, rotation=90, fontsize=8)
    plt.axvline(x=13.5, color='red', linestyle='--', label='Template | Global Features')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE - All visualizations generated!")
    print("="*60)
    print("\nKey observations to look for:")
    print("1. Are template matching scores high for correct ranks?")
    print("2. Is EMPTY class clearly separated in global features?")
    print("3. Which features show the most variance?")
    print("4. Are there strong correlations between certain features?")
    print("\nClose all plot windows to exit.")
    print("="*60 + "\n")
    
    plt.show(block=True)