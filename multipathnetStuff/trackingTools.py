import random

class MaxBipartite(object):

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def bpm(self, bpGraph, u, seen, matchR):

        for v in range(self.N):

            if bpGraph[u][v] and not seen[v]:
                seen[v] = True

                if (matchR[v] < 0 or self.bpm(bpGraph, matchR[v], seen, matchR)):
                    matchR[v] = u
                    return True

        return False

    def maxBPM(self, bpGraph):
        matchR = [0] * self.N

        for i in range(self.N):
            matchR[i] = -1

        result = 0
        for u in range(self.M):

            seen = [False] * self.N
            for i in range(self.N):
                seen[i] = False

            if self.bpm(bpGraph, u, seen, matchR):
                result += 1

        return result

def rand_array(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]


def main():
    bpGraph = [[False, True, True, False, False, False],
    [True, False, False, True, False, False],
    [False, False, True, False, False, False],
    [False, False, True, True, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, True]]

    m = MaxBipartite(6, 6)

    a = rand_array(6)
    print(a)
    print(m.maxBPM(a))

main()
