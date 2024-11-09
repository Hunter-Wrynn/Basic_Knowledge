import heapq

# 定义移动方向：上，下，左，右
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_STRINGS = ['U', 'D', 'L', 'R']

# 目标状态
GOAL_STATE = [1, 2, 3, 8, 0, 4, 7, 6, 5]

# 判断是否达到目标状态
def is_goal(state):
    return state == GOAL_STATE

# 打印棋盘状态
def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])

# 计算错位数（作为启发式函数）
def misplaced_tiles(state):
    distance = 0
    for i in range(9):
        if state[i] != 0 and state[i] != GOAL_STATE[i]:
            distance += 1
    return distance

# A* 算法实现
def a_star(start):
    start_pos = start.index(0)  # 找到初始状态中空白位置
    
    # 使用堆队列（优先队列）存储节点：f(n), g(n), state, zero_pos, path
    pq = []
    heapq.heappush(pq, (misplaced_tiles(start), 0, start, start_pos, ""))
    
    # 记录已访问的状态
    visited = set()
    visited.add(tuple(start))

    while pq:
        f, g, board, zero_pos, path = heapq.heappop(pq)  # 弹出 f(n) 最小的状态

        if is_goal(board):  # 如果当前状态是目标状态，输出结果
            print("Solution found!")
            print("Moves:", path)
            print_board(board)
            return

        # 计算空白位置的行列
        zero_row, zero_col = divmod(zero_pos, 3)

        # 生成新状态
        for i, (dx, dy) in enumerate(DIRECTIONS):
            new_row, new_col = zero_row + dx, zero_col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:  # 确保新位置合法
                new_zero_pos = new_row * 3 + new_col
                new_board = board[:]
                # 交换空白块与目标位置块
                new_board[zero_pos], new_board[new_zero_pos] = new_board[new_zero_pos], new_board[zero_pos]

                if tuple(new_board) not in visited:
                    visited.add(tuple(new_board))
                    new_g = g + 1  # 更新 g(n)，即步数加 1
                    new_h = misplaced_tiles(new_board)  # 计算新的启发式值（错位数）
                    # 将新的状态加入优先队列
                    heapq.heappush(pq, (new_g + new_h, new_g, new_board, new_zero_pos, path + DIR_STRINGS[i]))

    print("No solution found.")  # 如果队列为空，说明没有找到解

if __name__ == "__main__":
    start = [2, 0, 3, 1, 8, 4, 7, 6, 5]  # 输入的初始状态
    a_star(start)
