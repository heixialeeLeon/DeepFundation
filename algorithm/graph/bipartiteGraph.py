import queue

class Graph():
    def __init__(self, V):
        self.V = V
        self.graph = [[0 for column in range(V)] for row in range(V)]

    def isBipartite(self, init):
        color = [-1] * self.V
        color[init] = 1   # 1 for red, 0 for black
        que = queue.Queue()
        que.put(init)
        while not que.empty() :
            u = que.get()
            if self.graph[u][u] == 1:
                return False
            for v in range(self.V):
                if self.graph[u][v] == 1 and color[v] == -1:
                    color[v] = 1 - color[u]  # color the opposite color
                    que.put(v)
                if self.graph[u][v] == 1 and color[u] == color[v]:
                    return False
        return True


if __name__ == "__main__":
    g = Graph(4)
    g.graph = [[0, 1, 0, 1],
               [1, 0, 1, 0],
               [0, 1, 0, 1],
               [1, 0, 1, 0]
               ]
    print(g.isBipartite(0))