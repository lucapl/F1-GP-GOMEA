import itertools
import pickle
import random
from typing import TYPE_CHECKING

from deap import base, creator, tools

from src.linkage import LinkageModel, LinkageTreeFramsF1
from src.utils.stopping import EarlyStopper

if TYPE_CHECKING:  # circular import
    from src.our_toolbox import OurToolbox


# def flatten_dict(d):  # Similar to pd.json_normalize()
#     r = {}
#     for k, v in d.items():
#         for k2, v2 in v.items():
#             r[k + "." + k2] = v2
#     return r


def load_checkpoint(checkpoint):
    with open(checkpoint, "rb") as cp_file:
        cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    return population, start_gen, halloffame, logbook


def eaGOMEA(
    population: list,
    toolbox: "OurToolbox",  # base.Toolbox,
    start_gen=0,
    stats=None,
    halloffame=None,
    logbook: tools.Logbook | None = None,
    checkpoint_freq=None,
    checkpoint_name="./checkpoint_gomea_{gen}.pkl",
    verbose=False,
    fmut=10,
) -> tuple[list, tools.Logbook, list[list]]:
    """
    Tudelft GPGOMEA paper algorithm using deap

    toolbox funcs to register:
    - genepool_optimal_mixing
    - evaluate
    - build_linkage_model
    - should_stop - main loop stopping condition
    """
    # stats for observation
    if not logbook:
        logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    # evaluate not evaluated individuals
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(population) if stats else {}
    # from rich import print as rprint
    # rprint("First Multistats record: ", record)
    if start_gen not in logbook.select("gen"):
        # logbook.record(gen=start_gen, nevals=len(invalid_ind), **flatten_dict(record))
        logbook.record(gen=start_gen, nevals=len(invalid_ind), **record)
    linkage_log = []
    linkage_log.append([])  # empty for index 0 - match Logbook

    if halloffame is not None:
        halloffame.update(population)

    mutation_check = EarlyStopper(fmut, toolbox)

    # if verbose:
    #     print(logbook.stream)  # maybe won't shift??

    # main algorithm
    gen = start_gen + 1
    while not toolbox.should_stop(population, gen):
        # algorithm operators
        linkage_model: LinkageTreeFramsF1 = toolbox.build_linkage_model(population)
        lms, lpops = (
            [linkage_model for _ in range(len(population))],
            [population for _ in range(len(population))],
        )
        # toolboxes = [toolbox for _ in range(len(population))]
        new_pop = list(
            toolbox.map(toolbox.genepool_optimal_mixing, population, lms, lpops)
        )
        # print(new_pop[0])
        if mutation_check.shouldStop(new_pop):
            print("Mutation time!")
            new_pop = list(toolbox.map(toolbox.mutate, new_pop))
            invalid_ind = [ind for ind in new_pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        # print(new_pop[0])

        nevals = toolbox.get_evaluations()
        population = new_pop

        # logging and checkpointing further...
        if halloffame is not None:
            halloffame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        # from rich import print as rprint
        # rprint("Multistats record: ", record)
        # logbook.record(gen=gen, nevals=nevals, **flatten_dict(record))
        # if False:
        # record |= linkage_model.get_stats()
        current_linkage = linkage_model.get_stats()
        # rprint("Linkage_tree: ", current_linkage['linkage_tree'])
        linkage_log.append(current_linkage["linkage_tree"])

        if verbose:
            print(logbook.stream)

        if checkpoint_freq is not None and gen % checkpoint_freq == 0:
            cp = dict(
                population=population,
                generation=gen,
                halloffame=halloffame,
                logbook=logbook,
                rndstate=random.getstate(),
            )

            with open(checkpoint_name.format(gen=gen), "wb") as cp_file:
                pickle.dump(cp, cp_file)
        gen += 1
    # toolbox.mutate(population[0])
    # print(type(population), str(population[0]))
    return population, logbook, linkage_log


def gom(
    individual: list,
    linkage_model: LinkageModel,
    population: list[list[str]],
    toolbox: base.Toolbox,
    forcedImprov=True,
):
    """
    genepool optimal mixing operator from Tudelft GPGOMEA paper by Marco Virgolin

    toolbox funcs to register:
    - override_nodes
    - evaluate
    - forced_improvement (if used)

    produces indviduals as fit as the parents by mixing their genes
    """
    improving_ind = toolbox.clone(individual)
    cloned_ind = toolbox.clone(improving_ind)

    improvement = False

    for i in range(len(linkage_model)):
        # select random donor and linkage
        donor = tools.selRandom(population, 1)[0]
        f_i = linkage_model[donor]

        if f_i is None:
            continue
        # apply and evaluate
        improving_ind = toolbox.override_nodes(improving_ind, donor, f_i)

        if improving_ind == donor:
            continue

        o_fitness = toolbox.evaluate(improving_ind)
        improving_ind.fitness.values = o_fitness

        # assumes maximizing fitness; reverts worsening changes
        if improving_ind.fitness >= cloned_ind.fitness:
            cloned_ind = toolbox.clone(improving_ind)
            # cloned_ind = toolbox.override_nodes(cloned_ind, improving_ind, f_i)
            # cloned_ind.fitness = toolbox.clone(improving_ind.fitness)
            improvement = True
        else:
            improving_ind = toolbox.clone(cloned_ind)
            # improving_ind = toolbox.override_nodes(improving_ind, cloned_ind, f_i)
            # improving_ind.fitness = toolbox.clone(cloned_ind.fitness)

    if not improvement and forcedImprov:
        best = toolbox.get_best(population)#tools.selBest(population, 1)[0]
        if improving_ind != best:
            improving_ind = toolbox.forced_improvement(
                improving_ind, linkage_model, best
            )
    return improving_ind


def forced_improvement(improving_ind, linkage_model, donor, toolbox):
    """
    this probably speeds up convergendce very much
    """
    cloned_ind = toolbox.clone(improving_ind)

    for i in range(len(linkage_model)):
        f_i = linkage_model[donor]

        if f_i is None:
            continue

        if improving_ind == donor:
            continue

        # apply and evaluate
        improving_ind = toolbox.override_nodes(improving_ind, donor, f_i)

        o_fitness = toolbox.evaluate(improving_ind)
        improving_ind.fitness.values = o_fitness

        # assumes maximizing fitness; reverts worsening changes
        if improving_ind.fitness >= cloned_ind.fitness:
            return improving_ind  # end after improving
        else:
            improving_ind = toolbox.clone(cloned_ind)

    return toolbox.clone(donor)  # if nothing found, replaced with the best


def override_nodes(
    recipient: list, donor: list, f_i: tuple[int], toolbox, fillvalue="_"
):
    geno = [
        di if i in f_i else ri
        for i, (ri, di) in enumerate(
            itertools.zip_longest(recipient, donor, fillvalue=fillvalue)
        )
    ]
    geno = [i for i in geno if i != fillvalue]
    ind = creator.Individual(geno)
    ind.fitness = toolbox.clone(recipient.fitness)
    return ind
