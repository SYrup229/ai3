"""
Preprocessing Pipeline for Playing Card Rank Detection
Handles: Red background removal, card detection, ROI extraction
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class CardPreprocessor:
    """Preprocessing pipeline for playing card images"""
    
    def __init__(self, debug=False):
        """
        Initialize preprocessor
        
        Args:
            debug (bool): If True, shows intermediate processing steps
        """
        self.debug = debug
    
    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path (str or Path): Path to card image
            
        Returns:
            dict: Contains 'roi' (rank region), 'card' (full card), 'success' (bool)
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return {'success': False, 'error': 'Failed to load image'}
        
        if self.debug:
            self._show_debug("Original", img)
        
        # Step 1: Remove red background and detect card
        card_mask = self._remove_red_background(img)
        
        # Step 2: Find card contour
        card_contour = self._find_card_contour(card_mask)
        if card_contour is None:
            return {'success': False, 'error': 'Could not detect card'}
        
        # Step 3: Extract card region
        card_region = self._extract_card_region(img, card_contour)
        
        if self.debug:
            self._show_debug("Extracted Card", card_region)
        
        # Step 4: Extract rank ROI (top-left corner)
        rank_roi = self._extract_rank_roi(card_region)
        
        if rank_roi is None:
            return {'success': False, 'error': 'Could not extract rank ROI'}
        
        if self.debug:
            self._show_debug("Rank ROI", rank_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'success': True,
            'roi': rank_roi,
            'card': card_region,
            'mask': card_mask
        }
    
    def _remove_red_background(self, img):
        """
        Remove red background and create binary mask of card
        
        Args:
            img: BGR image
            
        Returns:
            Binary mask where card is white, background is black
        """
        # Enhance brightness to help separate card from noise
        # Convert to HSV and increase V channel
        hsv_bright = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_bright)
        
        # Increase brightness by 30% but cap at 255
        v = cv2.add(v, 50)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        # Merge back and convert to BGR
        hsv_bright = cv2.merge([h, s, v])
        img_enhanced = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
        
        if self.debug:
            self._show_debug("Brightness Enhanced", img_enhanced)
        
        # Apply bilateral filter to reduce noise while preserving edges
        img_filtered = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
        
        # Also convert to LAB color space for luminance normalization
        lab = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        # This helps normalize lighting variations
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back and convert to HSV
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        hsv_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2HSV)
        
        # Use normalized HSV for red detection with adjusted thresholds
        # Red color range in HSV (red wraps around, so two ranges)
        # Lower red range (0-10) - more permissive for lighting variations
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper red range (170-180)
        lower_red2 = np.array([165, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red on normalized image
        mask1 = cv2.inRange(hsv_normalized, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_normalized, lower_red2, upper_red2)
        
        # Also try on original HSV (for areas with good lighting)
        mask1_orig = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2_orig = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine all masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask_orig = cv2.bitwise_or(mask1_orig, mask2_orig)
        red_mask = cv2.bitwise_or(red_mask, red_mask_orig)
        
        # Card is NOT red, so invert
        card_mask = cv2.bitwise_not(red_mask)
        
        # Aggressive noise removal - multiple passes with different kernel sizes
        # First pass: remove small speckles
        kernel_small = np.ones((3, 3), np.uint8)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Second pass: close small gaps
        kernel_medium = np.ones((7, 7), np.uint8)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Third pass: remove remaining noise
        kernel_large = np.ones((11, 11), np.uint8)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel_large, iterations=1)
        
        # Final cleanup: close any remaining gaps
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        # Additional: Use median blur to remove salt-and-pepper noise
        card_mask = cv2.medianBlur(card_mask, 5)
        
        if self.debug:
            self._show_debug("Card Mask", card_mask)
        
        return card_mask
    
    def _find_card_contour(self, mask):
        """
        Find the card contour using multiple methods
        
        Args:
            mask: Binary mask
            
        Returns:
            Card contour or None if not found
        """
        # Try contour-based detection
        contour1, score1 = self._find_card_by_contours(mask)
        
        # Try edge-based detection
        contour2, score2 = self._find_card_by_edges(mask)
        
        if self.debug:
            print(f"\nContour approach score: {score1}")
            print(f"Edge approach score: {score2}")
        
        # Return the better result
        if score1 >= score2 and score1 >= 3:
            return contour1
        elif score2 >= 3:
            return contour2
        else:
            return None
    
    def _find_card_by_contours(self, mask):
        """
        Find card using contour detection
        
        Returns:
            (contour, score) tuple
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return (None, -1)
        
        img_h, img_w = mask.shape
        img_center = (img_w / 2, img_h / 2)
        img_area = img_h * img_w
        
        best_contour = None
        best_score = -1
        
        if self.debug:
            print(f"\n=== Analyzing {len(contours)} contours ===")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 0.03 * img_area:
                if self.debug:
                    print(f"Contour {i}: SKIPPED (too small, area={area:.0f})")
                continue
            
            # Skip very large contours
            if area > 0.9 * img_area:
                if self.debug:
                    print(f"Contour {i}: SKIPPED (too large, area={area:.0f})")
                continue
            
            # Get properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # Distance from center
            contour_center = (x + w/2, y + h/2)
            dx = contour_center[0] - img_center[0]
            dy = contour_center[1] - img_center[1]
            distance_from_center = np.sqrt(dx**2 + dy**2)
            max_distance = np.sqrt(img_w**2 + img_h**2)
            normalized_distance = distance_from_center / max_distance
            
            # Rectangularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                rectangularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                rectangularity = 0
            
            # Skip if not rectangular enough (very permissive threshold)
            # When card touches noise, rectangularity drops significantly
            if rectangularity < 0.10:
                if self.debug:
                    print(f"Contour {i}: SKIPPED (not rectangular, rect={rectangularity:.2f})")
                continue
            
            # Calculate score
            score = 0
            size_ratio = area / img_area
            
            if 0.1 < size_ratio < 0.6:
                score += 2
            if normalized_distance < 0.3:
                score += 3
            elif normalized_distance < 0.5:
                score += 1
            if 1.2 < aspect_ratio < 2.0:
                score += 2
            elif 1.0 < aspect_ratio < 2.5:
                score += 1
            
            # CRITICAL: Rectangularity is the most important factor for cards
            # Give much higher weight to rectangular shapes
            if rectangularity > 0.75:
                score += 10  # Very rectangular - huge bonus
            elif rectangularity > 0.6:
                score += 6   # Pretty rectangular - good bonus
            elif rectangularity > 0.4:
                score += 3   # Somewhat rectangular - moderate bonus
            elif rectangularity > 0.2:
                score += 1   # Barely rectangular - small bonus
            # Below 0.2: no bonus, and very low threshold means it barely qualifies
            
            if self.debug:
                print(f"Contour {i}: area={area:.0f} ({size_ratio:.2%}), "
                      f"aspect={aspect_ratio:.2f}, dist={normalized_distance:.2f}, "
                      f"rect={rectangularity:.2f}, score={score}")
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if self.debug:
            print(f"Best score: {best_score}, Required: 3")
        
        return (best_contour, best_score)
    
    def _find_card_by_edges(self, mask):
        """
        Find card using edge detection and Hough lines
        
        Returns:
            (contour, score) tuple
        """
        try:
            # Edge detection
            edges = cv2.Canny(mask, 50, 150)
            
            # Find lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                    minLineLength=100, maxLineGap=50)
            
            if lines is None or len(lines) < 4:
                return (None, -1)
            
            # Separate horizontal and vertical lines
            h_lines = []
            v_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 20 or angle > 160:
                    h_lines.append((y1 + y2) / 2)
                elif 70 < angle < 110:
                    v_lines.append((x1 + x2) / 2)
            
            if len(h_lines) < 2 or len(v_lines) < 2:
                return (None, -1)
            
            # Create rectangle
            y_top = int(min(h_lines))
            y_bottom = int(max(h_lines))
            x_left = int(min(v_lines))
            x_right = int(max(v_lines))
            
            rect_contour = np.array([
                [[x_left, y_top]],
                [[x_right, y_top]],
                [[x_right, y_bottom]],
                [[x_left, y_bottom]]
            ], dtype=np.int32)
            
            # Score it
            w = x_right - x_left
            h = y_bottom - y_top
            area = w * h
            
            img_h, img_w = mask.shape
            img_area = img_h * img_w
            
            if area < 0.03 * img_area or area > 0.9 * img_area:
                return (None, -1)
            
            aspect_ratio = h / w if w > 0 else 0
            score = 0
            
            if 0.1 < area / img_area < 0.6:
                score += 2
            if 1.2 < aspect_ratio < 2.0:
                score += 2
            
            return (rect_contour, score)
            
        except Exception:
            return (None, -1)
    
    def _extract_card_region(self, img, contour):
        """
        Extract the card region using bounding rectangle
        
        Args:
            img: Original image
            contour: Card contour
            
        Returns:
            Cropped card image
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Extract card
        card = img[y:y+h, x:x+w]
        
        return card
    
    def _extract_rank_roi(self, card_img):
        """
        Extract the rank symbol from top-left corner of card
        
        Args:
            card_img: Cropped card image
            
        Returns:
            Rank ROI image (grayscale, binary)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        if self.debug:
            self._show_debug("Card Grayscale", gray)
        
        # First, find where the actual card is (dark region)
        # Card is black, background is brighter
        _, card_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find the card contour in this mask
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the largest dark region (the card)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop to just the card (remove background)
            gray = gray[y:y+h, x:x+w]
            
            if self.debug:
                self._show_debug("Card Only (no background)", gray)
        
        # Now threshold to get text
        # Card is dark with bright text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if self.debug:
            self._show_debug("Binary Card", binary)
        
        # Extract top-left corner (where rank symbol is)
        h, w = binary.shape
        rank_h = int(h * 0.25)
        rank_w = int(w * 0.30)
        
        roi = binary[0:rank_h, 0:rank_w]
        
        if self.debug:
            self._show_debug("ROI", roi)
        
        # Check if ROI is valid
        if roi is None or roi.size == 0:
            return None
        
        # Check if ROI has actual content
        white_pixels = np.count_nonzero(roi)
        if white_pixels < 100:
            return None
        
        # Resize to standard size
        roi = cv2.resize(roi, (100, 150), interpolation=cv2.INTER_AREA)
        
        return roi
    
    def _crop_to_content(self, roi):
        """
        Crop ROI to actual content
        
        Args:
            roi: Binary ROI image
            
        Returns:
            Tightly cropped ROI
        """
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return roi
        
        # Get bounding box of all contours
        x_min, y_min = roi.shape[1], roi.shape[0]
        x_max, y_max = 0, 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(roi.shape[1], x_max + padding)
        y_max = min(roi.shape[0], y_max + padding)
        
        # Crop
        cropped = roi[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def _show_debug(self, title, img):
        """Show debug image"""
        plt.figure(figsize=(10, 6))
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


def test_preprocessing(image_path, debug=True):
    """Test preprocessing on a single image"""
    preprocessor = CardPreprocessor(debug=debug)
    result = preprocessor.preprocess(image_path)
    
    if result['success']:
        print("✓ Preprocessing successful!")
        print(f"ROI shape: {result['roi'].shape}")
        print(f"Card shape: {result['card'].shape}")
    else:
        print(f"✗ Preprocessing failed: {result['error']}")
    
    return result


def batch_preprocess(data_dir, output_dir, debug=False):
    """Preprocess all images in dataset"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = CardPreprocessor(debug=debug)
    
    stats = {'success': 0, 'failed': 0, 'failed_files': []}
    
    rank_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    
    for rank_folder in rank_folders:
        rank = rank_folder.name
        print(f"\nProcessing rank: {rank}")
        
        rank_output_dir = output_dir / rank
        rank_output_dir.mkdir(exist_ok=True)
        
        image_files = list(rank_folder.glob("*.png"))
        
        for img_file in image_files:
            result = preprocessor.preprocess(img_file)
            
            if result['success']:
                output_path = rank_output_dir / img_file.name
                cv2.imwrite(str(output_path), result['roi'])
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append(str(img_file))
                print(f"  Failed: {img_file.name} - {result['error']}")
        
        print(f"  Processed {len(image_files)} images")
    
    print("\n" + "="*50)
    print(f"Total successful: {stats['success']}")
    print(f"Total failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        print("\nFailed files:")
        for f in stats['failed_files'][:10]:  # Show first 10
            print(f"  - {f}")
        if len(stats['failed_files']) > 10:
            print(f"  ... and {len(stats['failed_files']) - 10} more")
    
    return stats


if __name__ == "__main__":
    print("Preprocessing module ready!")
    print("\nUsage:")
    print("1. Test single image: test_preprocessing('path/to/image.png', debug=True)")
    print("2. Process all images: batch_preprocess('data/raw', 'data/processed')")