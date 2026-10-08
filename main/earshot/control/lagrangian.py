class Lagrange:
    def __init__(self, cost_limit, lr=0.05, init=0.0, max_value=100.0, learn=True):
        self.cost_limit = cost_limit
        self.lr = lr
        self.value = init
        self.max_value = max_value
        self.learn = learn

    def update(self, mean_cost):
        if self.learn:
            self.value += self.lr * (mean_cost - self.cost_limit)
            self.value = min(max(self.value, 0.0), self.max_value)
        return self.value
