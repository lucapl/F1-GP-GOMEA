
class SaveBest:
    def __init__(self):
        self.best_ind = None

    def check(self, ind, values, toolbox):
        ind_clone = toolbox.clone(ind)
        ind_clone.fitness.values = values
        if self.best_ind is None or self.best_ind.fitness < ind_clone.fitness:
            self.best_ind = ind_clone

    def get_best(self, *args, **kwargs):
        return self.best_ind
