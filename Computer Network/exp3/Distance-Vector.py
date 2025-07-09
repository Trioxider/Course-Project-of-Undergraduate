inf = float('inf')
graph1 = [[0, 1, inf, 6, inf, inf],
          [1, 0, 3, 4, inf, inf],
          [inf, 3, 0, 2, 6, inf],
          [6, 4, 2, 0, 9, 2],
          [inf, inf, 6, 9, 0, inf],
          [inf, inf, inf, 3, inf, 0]]

graph = ['0', '1', '2', '3', '4', ' 5']


# 初始化路由表的函数
def init_routing_table(node, topology):
    # 创建一个空的路由表
    routing_table = {}
    # 遍历拓扑图中的每个节点
    for i in graph:
        int_i = int(i)
        # 将自己的距离设为0，下一跳设为None
        if i == node:
            routing_table[int_i] = (0, None)
        # 将直接相邻的节点的距离和下一跳设为对应的值
        elif topology[node][int(i)] != inf:
            routing_table[int_i] = (topology[node][int_i], int_i)
        # 将不直接相邻的节点的距离设为无穷大，下一跳设为None
        else:
            routing_table[int_i] = (inf, None)
    # 返回路由表
    return routing_table


# 更新路由表的函数
def update_routing_table(node, topology, routing_tables):
    # 获取当前节点的路由表
    routing_table = routing_tables[node]
    # 遍历拓扑图中的每个节点
    for t in graph:
        int_t = int(t)
        # 如果是自己或者不直接相邻的节点，跳过
        if t == node or topology[node][int_t] == inf:
            continue
        # 获取相邻节点的路由表
        neighbor_table = routing_tables[int_t]
        # 遍历相邻节点的路由表中的每个目的节点
        for next, value in neighbor_table.items():
            # 计算通过相邻节点到达目的节点的距离
            distance = topology[node][int_t] + value[0]
            # 如果这个距离小于当前路由表中的距离，更新路由表中的距离和下一跳
            if distance < routing_table[next][0]:
                routing_table[next] = (distance, t)
    # 返回更新后的路由表
    return routing_table


if __name__ == '__main__':
    # 创建一个空的字典，用来存储所有节点的路由表
    routing_tables = {}
    # 遍历拓扑图中的每个节点，初始化路由表
    for j in graph:
        routing_tables[int(j)] = init_routing_table(int(j), graph1)
    # 遍历拓扑图中的每个节点，更新路由表
    for i in graph:
        int_i = int(i)
        # 获取更新前的路由表
        old_table = routing_tables[int_i].copy()
        # 获取更新后的路由表
        new_table = update_routing_table(int_i, graph1, routing_tables)
    # 遍历拓扑图中的每个节点，打印路由表
    for s in graph:
        print(f"Node {s}'s routing table:")
        for hop, value in routing_tables[int(s)].items():
            print(f"Destination: {hop}, Distance: {value[0]}, Next hop: {value[1]}")
        print()
