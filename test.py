from itertools import product


def count_internal_nodes(tree):
    parents_to_children = {}
    for i, t in enumerate(tree):
        if t not in parents_to_children.keys():
            parents_to_children[t] = []
        parents_to_children[t].append(i)
    return sum([1 if (p > -1 and len(c) > 0) else 0 for p, c in parents_to_children.items()])


tree = [1, 3, 1, -1, 3]
print(count_internal_nodes(tree))  # should print 2

#

import numpy as np

"""
def get_minimum_connections(matrix):
    m=np.array(matrix)
    non0cols=(np.triu(m)[:-1, 1:].sum(axis=0)>0).sum()
    print("non zero cols ",non0cols)
    return len(matrix)-1==non0cols
"""

from collections import defaultdict
from itertools import permutations


def matrix_to_dict(matrix):
    redundant_map = {i: set() for i in range(len(matrix))}
    # redundant map of all direct connections
    possible_to_add = set()
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            if matrix[i][j]:
                redundant_map[i].add(j)
                redundant_map[j].add(i)
            else:
                possible_to_add.add((i, j))
    return redundant_map, possible_to_add


def find_path(graph, start_vertex, end_vertex, path=None):
    """ find a path from start_vertex to end_vertex
        in graph """
    if path is None:
        path = []
    graph = graph.copy()
    path = [path]+[start_vertex]
    if start_vertex == end_vertex:
        return path
    if start_vertex not in graph:
        return None
    for vertex in graph[start_vertex]:
        if vertex not in path:
            extended_path = find_path(graph, vertex, end_vertex, path)
            if extended_path:
                return extended_path
    return None

def get_minimum_connections(matrix):

    directs, missing = matrix_to_dict(matrix)
    # redundant map of existing direct connections

    m_solutions=[]
    for n in range(1, len(missing) + 1):
        for perm in permutations(missing, n):
            print(perm)
            m_new = matrix.copy()
            for n1, n2 in perm:
                matrix[n1][n2] = 1

    m = np.array(matrix)

    connections = np.stack(np.where(np.triu(m)[:-1, :])).T
    all_directs = np.stack(np.where(np.triu(np.ones_like(m))[:-1, :])).T

    edges = np.vectorize(lambda x: chr(65 + x))(connections)
    edges_all = np.vectorize(lambda x: chr(65 + x))(all_directs)

    graph = defaultdict(list)

    # Loop to iterate over every
    # edge of the graph
    for edge in edges:
        a, b = edge[0], edge[1]

        # Creating the graph
        # as adjacency list
        graph[a].append(b)
        graph[b].append(a)


matrix = \
    [
        [False, True, False, False, True],
        [True, False, False, False, False],
        [False, False, False, True, False],
        [False, False, True, False, False],
        [True, False, False, False, False]
    ]
print(get_minimum_connections(matrix))  # should print 1
