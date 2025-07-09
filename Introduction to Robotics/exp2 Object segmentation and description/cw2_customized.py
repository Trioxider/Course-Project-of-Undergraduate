import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2


# ===== Task 1: Grayscale Conversion and Thresholding =====
def gray_cal(img):
    # Calculate grayscale histogram
    gray_hist = np.zeros([256], np.uint64)
    for i in range(row):
        for j in range(col):
            gray_hist[img[i, j]] += 1
    return gray_hist


def ostu_thresh(img):
    # Compute grayscale histogram
    gray_hist = gray_cal(img)
    # Normalize histogram
    norm_hist = gray_hist / float(row * col)
    zero_cumu_moment = np.zeros([256], np.float32)
    one_cumu_moment = np.zeros([256], np.float32)
    # Compute zero-order and first-order cumulative moments
    for i in range(256):
        if i == 0:
            zero_cumu_moment[i] = norm_hist[i]
            one_cumu_moment[i] = 0
        else:
            zero_cumu_moment[i] = zero_cumu_moment[i - 1] + norm_hist[i]
            one_cumu_moment[i] = one_cumu_moment[i - 1] + i * norm_hist[i]
    # Compute between-class variance and find optimal threshold
    mean = one_cumu_moment[255]
    thresh = 0
    sigma = 0
    for i in range(256):
        if zero_cumu_moment[i] == 0 or zero_cumu_moment[i] == 1:
            sigma_tmp = 0
        else:
            sigma_tmp = (mean * zero_cumu_moment[i] - one_cumu_moment[i]) ** 2 / (
                    zero_cumu_moment[i] * (1 - zero_cumu_moment[i]))
        if sigma_tmp > sigma:
            sigma = sigma_tmp
            thresh = i

    # Apply threshold to create binary image
    thresh_img = img.copy()
    for i in range(row):
        for j in range(col):
            if thresh_img[i, j] > thresh:
                thresh_img[i, j] = 255
            else:
                thresh_img[i, j] = 0
    return thresh, thresh_img


def fill_holes(binary_img, max_hole_size=None):
    """Fill holes (background regions not connected to image border) in binary image.

    Args:
        binary_img (numpy.ndarray): Binary image with 0 (background) and 255 (foreground).
        max_hole_size (int, optional): Maximum size of holes to fill. If None, fill all holes.

    Returns:
        numpy.ndarray: Binary image with holes filled.
    """
    # Invert the binary image to treat holes as foreground
    binary_inv = 255 - binary_img
    # Label all connected regions in the inverted image
    labeled, num_regions = connected_components(binary_inv)

    height, width = binary_img.shape
    output = binary_img.copy()

    for region_id in range(1, num_regions + 1):
        # Extract coordinates of the current region
        coords = np.argwhere(labeled == region_id)
        # Check if the region touches the image border
        on_edge = False
        for y, x in coords:
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                on_edge = True
                break
        # Skip regions touching the border or exceeding size limit
        if on_edge:
            continue
        if max_hole_size is not None and coords.shape[0] > max_hole_size:
            continue
        # Fill the hole by setting pixels to foreground (255)
        output[coords[:, 0], coords[:, 1]] = 255

    return output


class UnionFind:
    """Disjoint-set (Union-Find) data structure for
    efficiently managing equivalence labels in connected component labeling"""

    def __init__(self):
        """Initialize parent and rank dictionaries to track set memberships and tree heights"""
        self.parent = {}  # Key: label, Value: parent label
        self.rank = {}  # Key: label, Value: tree depth rank

    def find(self, x):
        """Find root of the set using path compression heuristic"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Merge two sets using union by rank heuristic to keep tree depth minimal"""
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1


# ===== Task 2: Connected Component Labeling (Custom Implementation) =====
def connected_components(binary_img):
    """Custom implementation of connected component labeling using Union-Find"""
    height, width = binary_img.shape
    uf = UnionFind()
    labeled = np.zeros_like(binary_img, dtype=np.int32)
    current_label = 1

    # First Pass: Assign labels and manage equivalences
    for y in range(height):
        for x in range(width):
            if binary_img[y, x] == 0:
                continue
            neighbors = []
            if y > 0 and labeled[y - 1, x] != 0:
                neighbors.append(labeled[y - 1, x])
            if x > 0 and labeled[y, x - 1] != 0:
                neighbors.append(labeled[y, x - 1])
            if not neighbors:
                labeled[y, x] = current_label
                uf.parent[current_label] = current_label
                uf.rank[current_label] = 0
                current_label += 1
            else:
                min_label = min(neighbors)
                labeled[y, x] = min_label
                for n in neighbors:
                    if n != min_label:
                        uf.union(min_label, n)

    # Second Pass: Resolve label equivalences and renumber
    label_set = set(uf.find(l) for l in uf.parent.values())
    label_map = {old: new for new, old in enumerate(sorted(label_set), 1)}

    for y in range(height):
        for x in range(width):
            if labeled[y, x] != 0:
                labeled[y, x] = label_map[uf.find(labeled[y, x])]

    return labeled, len(label_map)


# ===== Task 3: Object Feature Calculation =====
def calculate_moments(region):
    """Compute central moments for object orientation and position"""
    y_coords, x_coords = np.where(region)
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    mu11 = np.mean((x_coords - x_mean) * (y_coords - y_mean))
    mu20 = np.mean((x_coords - x_mean) ** 2)
    mu02 = np.mean((y_coords - y_mean) ** 2)
    angle = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
    return (x_mean, y_mean), angle


def get_boundary(region):
    """Extract boundary coordinates of the region"""
    boundary = []
    for y in range(region.shape[0]):
        row_x = np.where(region[y, :])[0]
        if row_x.size > 0:
            boundary.append((row_x[0], y))
            boundary.append((row_x[-1], y))
    return boundary


# ===== New Function: Draw region features including second-order moments, ellipse, and bounding box =====
def draw_region_features(ax, region, obj_id, colors, length=50):
    """
    For a given region, draw the following on the provided matplotlib Axes:
      1. Bounding rectangle
      2. Major axis of second-order moment
      3. Covariance ellipse
    """
    # Compute centroid and principal axis
    center, angle = calculate_moments(region)
    # Pixel coordinates in the region
    coords = np.argwhere(region)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    color = colors[obj_id % len(colors)]

    # Bounding rectangle
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)

    # Principal axis of second-order moment
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    ax.plot([center[0] - dx, center[0] + dx], [center[1] - dy, center[1] + dy],
            color=color, linewidth=2)

    # Covariance ellipse
    xy_coords = coords[:, [1, 0]]
    cov = np.cov(xy_coords.T)
    vals, vecs = np.linalg.eigh(cov)
    width, height = 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0])
    angle_deg = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    ellipse = Ellipse(xy=center, width=width, height=height,
                      angle=angle_deg, fill=False,
                      edgecolor=color, linewidth=2)
    ax.add_patch(ellipse)


# ===== Main Execution Flow =====
if __name__ == '__main__':
    img = cv2.imread('biaoding.jpg', 0)  # Read image in grayscale
    row, col = img.shape[0], img.shape[1]
    thresh, binary = ostu_thresh(img)  # Apply Otsu thresholding
    binary = cv2.medianBlur(binary, 9)  # Remove noise

    binary_filled = fill_holes(binary, max_hole_size=500)  # Fill small holes
    labeled, num_features = connected_components(binary_filled)  # Label components

    # Visualization images in different procedures
    plt.figure(figsize=(10, 6))
    plt.subplot(221)
    plt.imshow(binary, cmap='gray')
    plt.title('binary image')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(binary_filled, cmap='gray')
    plt.title('filled binary image')
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(labeled, cmap='gray')
    plt.title('connected image')
    plt.axis('off')

    # Subplot 4: Color each connected component and overlay features
    plt.subplot(224)
    ax = plt.gca()
    # Color each connected component
    h, w = labeled.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    # Define RGB color table
    colors_rgb = [
        (255, 0, 0),   # red
        (0, 255, 0),   # green
        (0, 0, 255),   # blue
        (255, 255, 0)  # yellow
    ]
    colors = ['red', 'white', 'pink', 'purple', 'cyan']
    for obj_id in range(1, num_features+1):
        mask = (labeled == obj_id)
        if mask.sum() < 300:  # Filter too tiny objects
            continue
        colored[mask] = colors_rgb[obj_id % len(colors_rgb)]
    ax.imshow(colored)
    ax.axis('off')
    ax.set_title("Colored Connected Regions + Features")
    # Draw second-order moment axis, ellipse, and bounding box
    for obj_id in range(1, num_features+1):
        region = (labeled == obj_id)
        if region.sum() < 300:
            continue
        draw_region_features(ax, region, obj_id, colors, length=50)

    plt.tight_layout()
    plt.show()
