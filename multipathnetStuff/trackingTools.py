

class MaxBipartite(object):

    def __init__(self, M, N):
        self.M = M
        self.N = N

    def bpm(self, bpGraph, u, seen, matchR):
        bpGraph
        for v in range(self.N):
            print([u,v], min(bpGraph[u]),min([x[v] for x in bpGraph]))
            
            if ((bpGraph[u][v] <= min(bpGraph[u]) and (bpGraph[u][v] <= min([x[v] for x in bpGraph])))  and not (seen[v]) ):
                print("potential good match is ",[u,v], bpGraph[u][v])
            #if bpGraph[u][v] and not seen[v]:
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
        for i in range(len(matchR)):
            print(matchR[i])

        return result

def main():
    bpGraph = [[0.063, 0.976, 0.630, 0.798, 0.926, 0.447],
               [0.906, 0.017, 0.585, 0.122, 0.796, 0.398],
               [0.401, 0.239, 0.033, 0.629, 0.727, 0.354],
               [0.147, 0.254, 0.272, 0.067, 0.830, 0.679], 
               [0.063, 0.756, 0.134, 0.515, 0.053, 0.924], 
               [0.300, 0.844, 0.578, 0.825, 0.183, 0.028]]
    """bpGraph = [[0.3638903659471453, 0.9761139171410832, 0.3303319231563049],
               [0.9065725264825174, 0.1735533682269319, 0.3853397311047968],
               [0.40188120553635587, 0.23944120955902193, 0.9439235781363713],
               [0.1476368120205701, 0.25475687161080596, 0.2720203489219124], 
               [0.08528319155757547, 0.7564186590012879, 0.13476264281037809], 
               [0.3008867406705923, 0.8444700107777172, 0.5783045229376217]]"""

    """bpGraph = [[False, True, True, False, False, False],
    [True, False, False, True, False, False],
    [False, False, True, False, False, False],
    [False, False, True, True, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, True]]"""
    """bpGraph = [[True, False, False, False, False, False],
    [False, True, False, True, False, False],
    [False, True, False, False, False, False],
    [True, False, False, False, False, False],
    [True, False, True, False, False, False],
    [False, False, False, False, True, False]]"""
    
    print([x[4] for x in bpGraph])
    
    print(len(bpGraph),len(bpGraph[0]), "array shape")
    #m = MaxBipartite(6, 6)
    m = MaxBipartite(len(bpGraph),len(bpGraph[0]))
    print("Num of matches found", m.maxBPM(bpGraph))

main()
