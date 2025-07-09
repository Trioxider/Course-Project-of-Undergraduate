import numpy as np
from matplotlib import pyplot as plt
import cv2


def compute_norm_correlation(patch_left, patch_right):
    """计算归一化互相关系数"""
    mean_left = np.mean(patch_left)
    mean_right = np.mean(patch_right)

    patch_left_prime = patch_left - mean_left
    patch_right_prime = patch_right - mean_right

    numerator = np.sum(patch_left_prime * patch_right_prime)
    denominator = np.sqrt(np.sum(patch_left_prime ** 2) * np.sum(patch_right_prime ** 2))

    return numerator / denominator if denominator != 0 else 0

def compute_ccorr(patch_left, patch_right):
    """非归一化互相关匹配 (TM_CCORR)"""
    return np.sum(patch_left * patch_right)


def compute_ccoeff(patch_left, patch_right):
    """零均值互相关匹配 (TM_CCOEFF)"""
    mean_left = np.mean(patch_left)
    mean_right = np.mean(patch_right)
    return np.sum((patch_left - mean_left) * (patch_right - mean_right))


def compute_sqdiff(patch_left, patch_right):
    """平方差匹配 (TM_SQDIFF)"""
    return np.sum((patch_left - patch_right) ** 2)


def template_matching(img_left, img_right, window_size, metric, min_disp, max_disp):
    h, w = img_left.shape
    pad = window_size
    disparity_map = np.zeros((h, w), dtype=np.float32)

    # 边缘填充（与OpenCV一致）
    img_left_pad = np.pad(img_left.astype(np.float32), pad, mode='edge')
    img_right_pad = np.pad(img_right.astype(np.float32), pad, mode='edge')

    # 预计算右图积分图（加速用）
    right_integral = cv2.integral(img_right_pad)
    right_sq_integral = cv2.integral(img_right_pad ** 2)
    right_integral = right_integral.astype(np.float64)
    right_sq_integral = right_sq_integral.astype(np.float64)

    # 窗口面积
    win_area = (2 * pad + 1) ** 2

    for y in range(pad, h + pad):
        for x in range(pad, w + pad):
            left_roi = img_left_pad[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # 计算左图统计量
            if metric in ['CCOEFF', 'SQDIFF']:
                left_sum = np.sum(left_roi)
                left_sum_sq = np.sum(left_roi ** 2)
                left_mean = left_sum / win_area
                left_var = left_sum_sq / win_area - left_mean ** 2

            # 搜索范围
            x_min = max(pad, x - max_disp)
            x_max = min(w + pad - 1, x - min_disp)

            best_val = -np.inf if metric in ['CCORR', 'CCOEFF'] else np.inf
            best_x = x

            for xr in range(x_min, x_max + 1):
                right_roi = img_right_pad[y - pad:y + pad + 1, xr - pad:xr + pad + 1]

                # 计算匹配值
                if metric == 'CCORR':
                    score = compute_ccorr(left_roi, right_roi)
                elif metric == 'CCOEFF':
                    # 右图统计量（积分图加速）
                    sum_r = right_integral[y + pad + 1, xr + pad + 1] - right_integral[y - pad, xr + pad + 1] \
                            - right_integral[y + pad + 1, xr - pad] + right_integral[y - pad, xr - pad]
                    sum_sq_r = right_sq_integral[y + pad + 1, xr + pad + 1] - right_sq_integral[y - pad, xr + pad + 1] \
                               - right_sq_integral[y + pad + 1, xr - pad] + right_sq_integral[y - pad, xr - pad]
                    mean_r = sum_r / win_area
                    cov = np.sum(
                        left_roi * right_roi) - left_mean * sum_r - mean_r * left_sum + win_area * left_mean * mean_r
                    std_r = np.sqrt(sum_sq_r / win_area - mean_r ** 2)
                    score = cov / (np.sqrt(left_var) * std_r + 1e-10)
                elif metric == 'SQDIFF':
                    # 右图统计量
                    sum_r = right_integral[y + pad + 1, xr + pad + 1] - right_integral[y - pad, xr + pad + 1] \
                            - right_integral[y + pad + 1, xr - pad] + right_integral[y - pad, xr - pad]
                    sum_sq_r = right_sq_integral[y + pad + 1, xr + pad + 1] - right_sq_integral[y - pad, xr + pad + 1] \
                               - right_sq_integral[y + pad + 1, xr - pad] + right_sq_integral[y - pad, xr - pad]
                    cross = np.sum(left_roi * right_roi)
                    score = left_sum_sq + sum_sq_r - 2 * cross
                else:
                    raise ValueError("Unsupported metric: {}".format(metric))

                # 更新最佳匹配
                if (metric in ['CCORR', 'CCOEFF'] and score > best_val) or \
                        (metric == 'SQDIFF' and score < best_val):
                    best_val = score
                    best_x = xr

            disparity = (x - best_x)
            disparity_map[y - pad, x - pad] = disparity if min_disp <= disparity <= max_disp else 0

    return disparity_map


def depth_from_disparity(disparity_map, focal_length, baseline):
    depth_map = np.zeros(disparity_map.shape)
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            disparity = abs(disparity_map[i, j])
            if disparity == 0:
                depth_map[i, j] = 0
                continue
            depth = np.log(focal_length * baseline / disparity) / np.log(3810) * 255
            depth_map[i, j] = depth
    return depth_map


if __name__ == '__main__':
    imgL = cv2.imread('view1_1.png', 0)
    imgR = cv2.imread('view5_1.png', 0)
    min_disp = 0
    max_disp = 64

    # 使用自定义模板匹配
    disparity_map = template_matching(imgL, imgR, 5, 'CCOEFF', min_disp, max_disp)
    depth_map = depth_from_disparity(disparity_map, 1000, 100)

    plt.figure(figsize=(20, 20))
    plt.subplot(221), plt.imshow(imgR, cmap='gray')
    plt.title('Right Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(imgL, cmap='gray')
    plt.title('Left Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(disparity_map, cmap='gray')
    plt.title('Disparity Map'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(depth_map, cmap='gray')
    plt.title('Depth Map'), plt.xticks([]), plt.yticks([])
    plt.show()