import argparse
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
from deap import base, creator, gp, tools
from dotenv import load_dotenv

import framspy
from framspy.src.FramsticksLibCompetition import FramsticksLibCompetition
from src.gomea import eaGOMEA, forced_improvement, gom, override_nodes
from src.gpf1 import create_f1_pset, parse
from src.linkage import LinkageTreeFramsF1
from src.utils.fpcontrol import print_fenv_state, restore_fenv
from src.utils.stopping import EarlyStopper, earlyStoppingOrMaxIter


load_dotenv()
# default values for --framslib and --sim_location
ENV_FRAMSTICKS_DLL = os.getenv("FRAMSTICKS_DLL_PATH", "./Framsticks52")
# ENV_FRAMSPY_PATH = os.getenv("FRAMSPY_PATH", "./framspy")
ENV_FRAMSPY_PATH = os.getenv("FRAMSPY_PATH", str(framspy.__path__[0]))


def prepare_gomea_parser(parser):
    parser.add_argument('-n', '--ngen', default=15, type=int)
    parser.add_argument('-p', '--popsize', default=20, type=int)
    parser.add_argument('-e', '--early_stop', default=10, type=int)
    parser.add_argument('-g', '--initial_geno_mutations', default=100, type=int)
    parser.add_argument('--parts', nargs='+', default=[20, 30], help='Initial genotypes parts range')
    parser.add_argument('--neurons', nargs='+', default=[6, 8], help='Initial genotypes neurons range')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--MOCK_EVALS', action="store_true")
    parser.add_argument('--sims',
                        nargs='+',
                        default=['eval-allcriteria.sim', 'recording-body-coords.sim'],
                        help='List of simulation files to use.')
    parser.add_argument('--no_forced_improv', action='store_true')
    parser.add_argument(
        "--sim_location",
        help="Specifies location of simfiles.",
        default=ENV_FRAMSPY_PATH,
        #  default="./framspy"
    )
    parser.add_argument(
        "--framslib",
        help="Specifies location of framstick engine.",
        default=ENV_FRAMSTICKS_DLL,
        #  default="./Framsticks52"
    )
    parser.add_argument("--pmut", help="Probability of mutation occuring", default=0.8, type=float)
    parser.add_argument('--fmut', help="Frequency of mutation occuring", default=10, type=int)
    parser.add_argument('--count_nevals', help="Counts evaluations of genotype", action="store_true")

    return parser

#original_control_word = save_fenv() # this is important to not cause floating point exceptions and others
#toolbox = LinkageToolbox('build_linkage_model', 'override_nodes')
#toolbox.define_default_linkages(CHARS, PREFIX, original_control_word)


solution_cache={}

def evaluate(ptree, pset, flib, invalid_fitness, criteria, mock_test=False):
    try:
        geno = str(gp.compile(ptree, pset))
    except:
        return (invalid_fitness,)
    if geno in solution_cache:
        return (solution_cache[geno],)
    geno = [geno]
    try:
        valid = flib.isValidCreature(geno)[0]
    except Exception:
        print(geno)
        raise Exception
    if not valid:
        return (invalid_fitness,)
    # before running a creature through a simulation we ensure the genotype is valid
    if not mock_test:
        value = flib.evaluate(geno)[0]#["evaluations"][''][criteria]
    else:
        value = random.expovariate()
    solution_cache[geno[0]] = value
    return (value, ) 

def mutate(individual, pset, pmut, toolbox):
    if np.random.random() >= pmut:
        return individual
    mutated = [str(gp.compile(individual, pset))]
    mutated = framsLib.mutate(mutated)
    mutated = parse(mutated[0].replace(" ",""), pset)
    ind = creator.Individual(mutated)
    # ind.fitness = toolbox.clone(individual.fitness)
    return ind


def generate_random(flib, parts: tuple[int, int], neurons: tuple[int, int], iters: int, geno_format="1"):
    return flib.getRandomGenotype(flib.getSimplest(geno_format), *parts, *neurons, iters, return_even_if_failed=True)


def create_ind(flib, pset, n=100):
    return parse(generate_random(flib, n)[0].replace(" ", ""), pset)


def create_subtree(flib, pset, low=0, high=100, type_=None):
    n = np.random.randint(low, high)
    return parse(generate_random(flib, n)[0].replace(" ", ""), pset)


def main():
    # prepare arguments
    parser = argparse.ArgumentParser(
                    prog='GOMEA experiment',
                    description='runs a gomea experiment on f1 framsticks population',
                    epilog='glhf')
    parser = prepare_gomea_parser(parser)
    args = parser.parse_args()

    ##########################
    # load Framsticks library
    ##########################
    # sim files
    frasmpy_path = Path(args.sim_location)
    sim_formatted = ';'.join([
        (frasmpy_path/sim_file).absolute().as_posix()
        for sim_file in args.sims
    ])

    # engine
    # print_fenv_state("Before loading framsticks")
    framsLib = FramsticksLibCompetition(args.framslib, None, sim_formatted)
    # print_fenv_state("After loading framsticks")
    # restore_fenv(original_control_word)

    #####################
    # deap definitions
    #####################
    pset = create_f1_pset()
    # framsticks classes
    creator.create("FitnessMax", base.Fitness, weights=[1])
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    early_stopper = EarlyStopper(args.early_stop, toolbox)
    # basic operators
    toolbox.register("random_individual", create_ind, flib=framsLib, pset=pset, iters=args.initial_geno_mutations, parts=args.parts, neurons=args.neurons)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, args.popsize)

    # evaluation for testing
    max_len = args.initial_geno_mutations
    toolbox.register("evaluate", evaluate, pset=pset, flib=framsLib, invalid_fitness=-999999.0, criteria="distance", mock_test=args.MOCK_EVALS)

    # early stopper or max itera
    toolbox.register("should_stop", partial(earlyStoppingOrMaxIter, max_gen=args.ngen, early_stopper=EarlyStopper(args.early_stop, toolbox)))

    # gomea operators
    isForcedImprov = not args.no_forced_improv
    print("Is forced improvent on?: ", isForcedImprov)
    toolbox.register("genepool_optimal_mixing", gom, toolbox=toolbox, forcedImprov=isForcedImprov)
    toolbox.register("forced_improvement", forced_improvement, toolbox=toolbox)
    toolbox.register("build_linkage_model", LinkageTreeFramsF1, original_control_word=None)
    toolbox.register("override_nodes", override_nodes, fillvalue="_", toolbox=toolbox)
    toolbox.register("mutate",mutate, pset=pset, pmut=args.pmut, toolbox=toolbox)
    toolbox.register("get_evaluations", framsLib.get_evals if args.count_nevals else lambda: 0)

    ####################
    # stats logging
    ####################
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)

    # stats_geno = tools.Statistics(lambda ind: ind[0])
    # stats_geno.register("entropy", calc_entropy)
    # stats_geno.register("gene_diversity", calc_gene_diversity, prefix_len=PREFIX, control_word=original_control_word)

    stats_geno_len = tools.Statistics(lambda ind: len(ind))
    stats_geno_len.register("avg", np.mean)
    stats_geno_len.register("std", np.std)
    stats_geno_len.register("min", np.min)
    stats_geno_len.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, genotype_length=stats_geno_len)

    #####################
    # population initialization
    #####################
    hof = None
    logbook = None
    start_gen = 0

    pop = toolbox.population()

    ########################
    # MAIN ALGORITHM
    ########################
    try:
        new_pop, logbook = eaGOMEA(pop, toolbox,
        start_gen=start_gen,
        logbook=logbook,
        stats=mstats,
        halloffame=hof,
        fmut = args.fmut,
        # checkpoint_freq=args.checkpoint_frequency,
        # checkpoint_name=checkpoints_out,
        verbose=args.verbose)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    #######################
    # saving outputs
    #######################
    framsLib.end()


if __name__ == "__main__":
    main()
