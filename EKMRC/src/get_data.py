import heapq


class TopK(object):
    """Top k largest"""
    def __init__(self, iterable, k):
        self.maxheap = []
        self.capacity = k
        self.iterable = iterable

    def push(self, val):
        "add val to heap"
        if len(self.maxheap) >= self.capacity:
            max_val = self.maxheap[0]
            if val < max_val: 
                pass
            else:
                heapq.heapreplace(self.maxheap, val)  
        else:
            heapq.heappush(self.maxheap, val)

    def get_topk(self):
        for val in self.iterable:
            self.push(val)
        return self.maxheap




import random
i = list(range(1000))  
random.shuffle(i)
_ = TopK(i, 10)
print(_.get_topk())

_.push(997)
print(_.get_topk())