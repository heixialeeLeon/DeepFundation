def remove_zero(graph):
    while 0 in graph:
        graph.remove(0)

def havel_hakimi(graph):
    if len(graph) == 0:
        return list()
    graph.sort(reverse=True)
    max_val = graph[0]
    if(len(graph)==1):
        return(graph)
    for index in range(1, len(graph)):
        if max_val > 0:
            max_val = max_val -1
            graph[0] = max_val
            if(max_val >= 0):
                graph[index] = graph[index]- 1
    remove_zero(graph)
    havel_hakimi(graph)
    return graph

if __name__ == "__main__":
    graph = [2, 2, 2, 2, 3]
    print(havel_hakimi(graph))