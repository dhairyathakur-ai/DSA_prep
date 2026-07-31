# K-Nearest Neighbors Classification

import heapq
from collections import Counter
from typing import List

class Solution:
    def knnClassify(self, points: List[List[int]], labels: List[int], 
                     queries: List[List[int]], k: int) -> List[int]:
        
        def squared_dist(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        
        answer = []
        
        for qx, qy in queries:
            # Max-heap of size k, storing (-dist, -index)
            # heap[0] = current "worst" neighbor:
            #   largest distance, and on tie, largest index
            heap = []
            
            for i, (px, py) in enumerate(points):
                d = squared_dist((px, py), (qx, qy))
                candidate = (-d, -i)
                
                if len(heap) < k:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    # candidate is closer, or same distance but smaller index
                    heapq.heapreplace(heap, candidate)
            
            # Tally labels among the k nearest neighbors
            freq = Counter(labels[-idx] for _, idx in heap)
            
            # Pick label with max frequency; tie -> smallest label
            best_label = min(freq.keys(), key=lambda lbl: (-freq[lbl], lbl))
            answer.append(best_label)
        
        return answer