import deap


def maxIterStop(population, gen, max_gen):
    return gen > max_gen


class EarlyStopper():
    def __init__(self, iters, toolbox):
        self.fitness = deap.creator.FitnessMax()
        self.fitness.values = (float('-inf'),)
        self.toolbox = toolbox
        self.i = 0
        self.iters = iters
    
    def _select_value(self, population):
        return self.toolbox.clone(deap.tools.selBest(population, 1)[0].fitness)

    def shouldStop(self, population):
        self.i += 1
        current_fitness = self._select_value(population)
        if current_fitness > self.fitness:
            self.i = 0
            self.fitness = current_fitness
        return self.i >= self.iters


def earlyStoppingOrMaxIter(population, gen, max_gen, early_stopper):
    return early_stopper.shouldStop(population) or maxIterStop(population, gen, max_gen)
