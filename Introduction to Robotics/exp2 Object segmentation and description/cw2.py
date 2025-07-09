import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# 1. 将图像转换为灰度并阈值化以创建二值图像
def gray_cal(img):
    gray_hist = np.zeros([256], np.uint64)
    for i in range(row):
        for j in range(col):
            gray_hist[img[i, j]] += 1
    return gray_hist


def ostu_thresh(img):
    # 计算灰度直方图
    gray_hist = gray_cal(img)
    # 将灰度直方图归一化
    norm_hist = gray_hist / float(row * col)
    zero_cumu_moment = np.zeros([256], np.float32)
    one_cumu_moment = np.zeros([256], np.float32)
    # 计算零阶和一阶累积矩
    for i in range(256):
        if i == 0:
            zero_cumu_moment[i] = norm_hist[i]
            one_cumu_moment[i] = 0
        else:
            zero_cumu_moment[i] = zero_cumu_moment[i - 1] + norm_hist[i]
            one_cumu_moment[i] = one_cumu_moment[i - 1] + i * norm_hist[i]
    # 计算方差，找到最大方差对应阈值
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

    thresh_img = img.copy()
    for i in range(row):
        for j in range(col):
            if thresh_img[i, j] > thresh:
                thresh_img[i, j] = 255
            else:
                thresh_img[i, j] = 0
    return thresh, thresh_img


def connected_components(binary):
    label = np.zeros((row, col), dtype=int)
    flag = 1
    root = {}
    # 定义并查集的数据结构，用一个数组来存储每个元素的父节点
    parent = [i for i in range(row * col)]

    # 定义并查集的find操作，用于查找一个元素的根节点，并进行路径压缩
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root != y_root:
            parent[x_root] = y_root

    # 第一遍扫描
    for i in range(row):
        for j in range(col):
            if binary[i, j] != 0:
                left = label[i, j - 1] if j > 0 else 0
                up = label[i - 1, j] if i > 0 else 0
                if left == 0 and up == 0:
                    label[i, j] = flag
                    # equiv_table[flag] = flag # 初始化等价表
                    flag += 1
                elif left != 0 or up != 0:
                    if left == 0 or up == 0:
                        label[i, j] = max(left, up)
                    else:
                        label[i, j] = min(left, up)
                        union(max(left, up), min(left, up))

    # 对图像的每个像素进行扫描
    for i in range(row):
        for j in range(col):
            # 如果像素值不为0，就将它的标签替换为它在并查集中的根节点
            if label[i, j] != 0:
                label[i, j] = find(label[i, j])

    label = label.astype(np.float32)  # 转换为浮点数
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(label)
    label = (label - min_val) / (max_val - min_val)  # 归一化到0到1之间
    label = label * 255  # 乘以255得到灰度值
    label = label.astype(np.uint8)  # 转换为无符号整数

    label_count = 0
    for i in range(row):
        for j in range(col):
            if label[i, j] >= label_count:
                label_count = label[i, j]

    return label_count, label


# 3. 对于每个对象
def process_objects(binary_image, num_labels, labels_im):
    _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    color_map = np.random.randint(low=0, high=256, size=(num_labels + 1, 3))
    for label in range(1, num_labels + 1):
        # a. 计算并绘制其最小二阶矩的轴（方向方程）
        # b. 找到其边界
        # c. 找到并绘制其边界框
        # d. 找到并绘制最佳逼近对象形状的椭圆的方程
        color_image[labels_im == label] = color_map[label]

        for contour in contours:
            # a. 计算并绘制其最小二阶矩的轴（方向方程）
            # 计算轮廓的中心和方向角
            moment = cv2.moments(contour)
            # 计算物体的质心
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            # 计算最小二阶矩
            mu20 = moment['mu20'] / moment['m00']
            mu11 = moment['mu11'] / moment['m00']
            mu02 = moment['mu02'] / moment['m00']
            # 计算最小特征值
            lambda_min = (mu20 + mu02 - math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)) / 2
            # 使用中心二阶矩，计算图像的方向
            theta = math.atan2(2 * mu11, mu20 - mu02) / 2

            # 计算轮廓的主轴长度并绘制方向线
            x1 = int(cx + math.sqrt(lambda_min) * math.cos(theta))
            y1 = int(cy + math.sqrt(lambda_min) * math.sin(theta))
            x2 = int(cx - math.sqrt(lambda_min) * math.cos(theta))
            y2 = int(cy - math.sqrt(lambda_min) * math.sin(theta))
            cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # b. 找到其边界
            x, y, w, h = cv2.boundingRect(contour)

            # c. 找到并绘制其边界框
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # d. 找到并绘制最佳逼近对象形状的椭圆的方程
            if len(contour) >= 5:  # fitEllipse需要至少5个点
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(color_image, ellipse, (255, 0, 255), 2)

    plt.imshow(color_image, cmap='jet')
    plt.title('Color image')
    plt.show()


# 使用上述函数
img1 = cv2.imread('Resources/biaoding.jpg', 0)
row = img1.shape[0]
col = img1.shape[1]
thresh, binary_image = ostu_thresh(img1)
binary_image = cv2.medianBlur(binary_image, 7)

plt.imshow(binary_image, cmap='gray')
plt.title('Binary image')
plt.show()

num_labels, labels_im = connected_components(binary_image)
plt.imshow(labels_im, cmap='gray')
plt.title('label image')
plt.show()

process_objects(binary_image, num_labels, labels_im)
