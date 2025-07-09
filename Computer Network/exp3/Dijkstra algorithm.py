# 定义一个函数 dijkstra，用于计算从一个节点到其他节点的最短路径
def dijkstra(graph1, root):
    # 定义一个无穷大的常量
    inf = float('inf')
    # 定义一个字典 distance1，用于存储每个节点到根节点的距离
    distance1 = {}
    # 定义一个字典 predecessor1，用于存储每个节点的前驱节点
    predecessor1 = {}
    # 定义一个集合 visited，用于存储已经访问过的节点
    visited = set()
    # 遍历图中的所有节点
    for N in graph1:
        # 如果节点是根节点，那么它到自己的距离为0
        if N == root:
            distance1[N] = 0
        # 否则，它到根节点的距离为无穷大
        else:
            distance1[N] = inf
        # 初始化每个节点的前驱节点为 None
        predecessor1[N] = None

    # 定义一个内部函数 get_distance，用于获取一个节点到根节点的距离，如果该节点已经访问过，那么返回无穷大
    def get_distance(x):
        if x not in visited:
            return distance1[x]
        else:
            return inf

    # 当访问过的节点数量小于图中的节点数量时，循环执行以下操作
    while len(visited) < len(graph):
        # 初始化一个最小键和一个最小值
        min_key = None
        min_value = inf
        # 遍历 distance1 中的所有键
        for key in distance1:
            # 获取该键对应的值，即节点到根节点的距离
            value = get_distance(key)
            # 如果该值小于最小值，那么更新最小键和最小值
            if value < min_value:
                min_value = value
                min_key = key
        # 将最小键赋值给 current，表示当前要访问的节点
        current = min_key
        # 将 current 加入到 visited 集合中，表示已经访问过
        visited.add(current)
        # 遍历 current 的所有邻居节点和对应的权重
        for neighbour, weight in graph[current]:
            # 计算从 current 到邻居节点的距离
            new_distance = distance1[current] + weight
            # 如果该距离小于邻居节点到根节点的距离，那么更新 distance1 和 predecessor1
            if new_distance < distance1[neighbour]:
                distance1[neighbour] = new_distance
                predecessor1[neighbour] = current
    # 返回 distance1 和 predecessor1
    return distance1, predecessor1


# 定义一个函数 get_path，用于根据 predecessor1 获取从根节点到终点的最短路径
def get_path(predecessor1, root, end):
    # 定义一个空列表 path_，用于存储路径上的节点
    path_ = []
    # 从终点开始，沿着前驱节点回溯
    node = end
    while node != root:
        # 将节点插入到 path_ 的开头
        path_.insert(0, node)
        # 更新节点为其前驱节点
        node = predecessor1[node]
    # 将根节点也插入到 path_ 的开头
    path_.insert(0, root)
    # 返回 path_
    return path_


# 如果当前文件是主文件，那么执行以下操作
if __name__ == '__main__':
    # 定义一个图，用字典表示，键是节点，值是邻居节点和权重的列表
    graph = {'0': [('1', 1), ('3', 6)],
             '1': [('0', 1), ('2', 3), ('3', 4)],
             '2': [('1', 3), ('3', 2), ('4', 6)],
             '3': [('0', 6), ('1', 4), ('2', 2), ('4', 9), ('5', 2)],
             '4': [('2', 6), ('3', 9)],
             '5': [('3', 3)]}

    # 遍历图中的所有节点
    for i in graph:
        # 调用 dijkstra 函数，传入图和节点 i 作为参数，返回 distance 和 predecessor
        distance, predecessor = dijkstra(graph, i)
        # 打印从节点 i 到其他节点的最短路径
        print(f"The least cost path from node {i} to other nodes is:")
        print(distance)
        # 打印每个节点的前驱节点
        print('The predecessor dictionary of each node is:')
        print(predecessor)

        # 遍历图中的所有节点
        for j in graph:
            # 如果节点 i 和 j 相同，那么跳过
            if i == j:
                continue
            # 否则，调用 get_path 函数，传入 predecessor，节点 i 和 j 作为参数，返回 path
            else:
                path = get_path(predecessor, i, j)
                # 打印从节点 i 到节点 j 的最短路径上的节点
                print(f'The detailed node path from the least cost path from node {i} to {j} is:')
                print(path)
