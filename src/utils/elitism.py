
class SaveBest:
    def __init__(self, toolbox):
        self.best_ind = None
        self.toolbox = toolbox

    def check(self, ind, values):
        ind_clone = self.toolbox.clone(ind)
        ind_clone.fitness.values = values
        if self.best_ind == None or self.best_ind.fitness < ind_clone.fitness:
            self.best_ind = ind_clone

    def get_best(self, *args, **kwargs):
        return self.best_ind
