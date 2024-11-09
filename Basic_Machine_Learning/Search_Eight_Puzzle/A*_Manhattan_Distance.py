import heapq

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
DIR_STRINGS = ['U', 'D', 'L', 'R']

# 目标状态
GOAL_STATE = [1, 2, 3, 8, 0, 4, 7, 6, 5]

def is_goal(state):
    return state == GOAL_STATE

def manhattan_distance(state):
    """ 计算曼哈顿距离 """
    distance = 0
    for i in range(9):
        if state[i] == 0:
            continue
        goal_pos = GOAL_STATE.index(state[i])
        distance += abs(i // 3 - goal_pos // 3) + abs(i % 3 - goal_pos % 3)
    return distance

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i:i+3])

def a_star(start):
    start_pos = start.index(0)
    
    # 用优先队列进行启发式搜索
    pq = []
    heapq.heappush(pq, (manhattan_distance(start), 0, start, start_pos, ""))  # f(n), g(n), state, empty_pos, path
    visited = set()
    visited.add(tuple(start))

    while pq:
        f, g, board, zero_pos, path = heapq.heappop(pq)  # 弹出最小的f(n)值的状态

        if is_goal(board):
            print("Solution found!")
            print("Moves:", path)
            print_board(board)
            return
        
        zero_row, zero_col = divmod(zero_pos, 3)

        for i, (dx, dy) in enumerate(DIRECTIONS):
            new_row, new_col = zero_row + dx, zero_col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_zero_pos = new_row * 3 + new_col
                new_board = board[:]
                new_board[zero_pos], new_board[new_zero_pos] = new_board[new_zero_pos], new_board[zero_pos]

                if tuple(new_board) not in visited:
                    visited.add(tuple(new_board))
                    new_g = g + 1  # 当前的路径长度加 1
                    new_h = manhattan_distance(new_board)  # 计算新的启发式估计值
                    heapq.heappush(pq, (new_g + new_h, new_g, new_board, new_zero_pos, path + DIR_STRINGS[i]))

    print("No solution found.")

if __name__ == "__main__":
    start = [2, 0, 3, 1, 8, 4, 7, 6, 5]  # 输入的初始状态
    a_star(start)
