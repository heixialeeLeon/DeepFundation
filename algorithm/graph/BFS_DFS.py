from collections import defaultdict
import queue

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFSUitl(self, v, visited):
        visited[v] = True
        print(v)

        for i in self.graph[v]:
            if visited[i] == False:
                self.DFSUitl(i, visited)

    def DFS(self):
        V= len(self.graph)
        visited =[False]*V
        for i in range(V):
            if visited[i] == False:
                self.DFSUitl(i,visited)

    def BFS(self,start):
        V = len(self.graph)
        visited = [False]*V
        record_queue = queue.Queue()
        record_queue.put(start)
        while not record_queue.empty():
            u = record_queue.get()
            if visited[u] == False:
                print(u)
                visited[u]=True
                for v in self.graph[u]:
                    record_queue.put(v)

if __name__ == "__main__":
    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 3)
    g.addEdge(1, 4)
    g.addEdge(1, 5)
    g.addEdge(4, 4)
    g.addEdge(5, 5)

    print("dfs travel")
    g.DFS()
    print("bfs travel")
    g.BFS(0)