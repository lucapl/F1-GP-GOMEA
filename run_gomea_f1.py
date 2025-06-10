import os, sys
import json
import math
import argparse
from pathlib import Path

from framspy.src.FramsticksLib import FramsticksLib
from framspy.src.FramsticksEvolution import save_genotypes

import numpy as np
from deap import creator, base, tools, gp

from src.gomea import load_checkpoint, eaGOMEA, gom, forced_improvement, override_nodes
from src.linkage import LinkageTreeFramsF1
from src.gpf1 import create_f1_pset

from src.stats import calc_uniqueness, calc_gene_diversity, calc_entropy

from src.utils.stopping import EarlyStopper
from src.utils.encoding import logbook_encoder, NpEncoder
from src.utils.fpcontrol import *

def prepare_gomea_parser(parser, linkage_choices):
    #parser.add_argument('-l', '--linkage', required=True, choices=linkage_choices)
    parser.add_argument('-n', '--ngen', default=250, type=int)
    parser.add_argument('-p', '--popsize', default=200, type=int)
    parser.add_argument('-e', '--early_stop', default=10, type=int)
    parser.add_argument('-g', '--geno_length', default=15, type=int)
    #parser.add_argument('--enforce_geno_len', action='store_true')
    #parser.add_argument('--save_evaluator', action='store_true')
    parser.add_argument('--hof_size', default=1, type=int)
    #parser.add_argument('-f', '--checkpoint_frequency', default=10, type=int)
    #parser.add_argument('-C', '--checkpoint') # checkpoint to load
    parser.add_argument('-v', '--verbose', action='store_true')
    #parser.add_argument('-M', '--multithread', action='store_true')
    #parser.add_argument('-O', '--output', default='./out/')
    parser.add_argument('-T', '--test_eval', action='store_true')
    parser.add_argument('--no_forced_improv', action='store_true')
    parser.add_argument('--pop_six', action='store_true')

    return parser

#original_control_word = save_fenv() # this is important to not cause floating point exceptions and others
#toolbox = LinkageToolbox('build_linkage_model', 'override_nodes')
#toolbox.define_default_linkages(CHARS, PREFIX, original_control_word)


def evaluate(ptree, pset, flib, invalid_fitness, criteria):
    try:
        geno = [gp.compile(ptree, pset)]
    except:
        return (invalid_fitness,)
    valid = flib.isValidCreature(geno)[0]
    if not valid:
        return (invalid_fitness,)
    # before running a creature through a simulation we ensure the genotype is valid
    value = flib.evaluate(geno)[0]["evaluations"][''][criteria]
    return (value,) 


def random_ind(geno, length):
    return creator.Individual([frams_random_genotype(geno,length)])


if __name__ == '__main__':
    # prepare arguments
    parser = argparse.ArgumentParser(
                    prog='GOMEA experiment',
                    description='runs a gomea experiment on f9 framsticks population',
                    epilog='glhf')
    parser = prepare_gomea_parser(parser, toolbox.get_registered_linkages())
    args = parser.parse_args()

    ##########################
    # load Framsticks library
    ##########################
    # sim files
    frasmpy_path = Path("./framspy/eval-allcriteria.sim").absolute().as_posix()
    sim_formatted = ';'.join([
        f'{os.environ["FRAMSPY_PATH"]}/{sim_file}'
        for sim_file in ['eval-allcriteria-mini.sim']
    ])

    # engine
    print_fenv_state("Before loading framsticks")
    framsLib = FramsticksLib(os.environ["FRAMSTICKS_ENGINE_PATH"], None, sim_formatted)
    print_fenv_state("After loading framsticks")
    restore_fenv(original_control_word)

    # framsticks evaluator
    evaluator = FramsEvaluator(
        framsLib,
        fake_eval=args.test_eval)  # fast fake evaluations for testing
    if args.test_eval:
        print("WARNING: Running fake evaluations intended for testing!")

    #####################
    # deap definitions
    #####################
    pset = create_f1_pset()
    # framsticks classes
    creator.create("FitnessMax", base.Fitness, weights=[1])
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    early_stopper = EarlyStopper(args.early_stop, toolbox)
    # basic operators
    toolbox.register("attr_simplest_genotype", frams_getsimplest, None, '9', None)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("random_individual", random_ind, '9')
    toolbox.register("one_letter_ind", one_letter_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation for testing
    max_len = args.geno_length
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.decorate("evaluate", LengthPenalty(
    lambda ind: len(ind[0])<=(max_len+PREFIX),
    (max_len if args.soft_len_dropoff else None,
        max_len if args.enforce_geno_len else None),
    PREFIX,
    args.soft_len_dropoff,
    lambda v, w: w * v**2))

    # early stopper or max itera
    toolbox.register("should_stop", earlyStoppingOrMaxIter, max_gen=args.ngen, early_stopper=early_stopper)

    # gomea operators
    isForcedImprov = not args.no_forced_improv
    print("Is forced improvent on?: ", isForcedImprov)
    toolbox.register("genepool_optimal_mixing",gom,toolbox=toolbox,forcedImprov=isForcedImprov)
    toolbox.register("forced_improvement",forced_improvement, toolbox=toolbox)

    toolbox.register("get_evaluations", get_evaluations, evaluator)

    # register linkage models
    toolbox.use_linkage(args.linkage)

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

    stats_geno_len = tools.Statistics(lambda ind: len(ind[0]))
    stats_geno_len.register("avg", np.mean)
    stats_geno_len.register("std", np.std)
    stats_geno_len.register("min", np.min)
    stats_geno_len.register("max", np.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, genotype=stats_geno, genotype_length=stats_geno_len)

    if args.checkpoint and args.output == 'same':
        args.output = os.path.dirname(args.checkpoint)

    #####################
    # population initialization
    # or loading from checkpoint
    #####################
    if args.checkpoint:
        pop, start_gen, hof, logbook = load_checkpoint(args.checkpoint)
    else:
        # hof
        hof = tools.HallOfFame(args.hof_size)
        logbook = None
        start_gen = 0

    if args.pop_six and not args.checkpoint:
        # pop 6 experiment
        pop = [toolbox.one_letter_ind(letter, args.geno_length) for letter in CHARS]
    elif not args.checkpoint:
        pop = [toolbox.random_individual(args.geno_length) for _ in range(args.popsize)]

    if args.output[-1] != '/': args.output += '/'
    checkpoints_out = args.output+'checkpoint_{gen}.pkl'

    os.makedirs(args.output, exist_ok=True)
    ########################
    # MAIN ALGORITHM
    ########################
    new_pop, logbook = eaGOMEA(pop, toolbox,
    start_gen=start_gen,
    logbook=logbook,
    stats=mstats,
    halloffame=hof,
    checkpoint_freq=args.checkpoint_frequency,
    checkpoint_name=checkpoints_out,
    verbose=args.verbose)

    #######################
    # saving outputs
    #######################
    with open(args.output+"logbook.json", 'w') as file:
        json.dump(logbook_encoder(logbook), file, cls=NpEncoder)

    with open(args.output+"final_pop.json", 'w') as file:
        json.dump(new_pop, file)

    if args.save_evaluator:
        with open(args.output+"evaluator.pkl", "w") as file:
            json.dump(evaluator.evaluated_genos, file)

    save_genotypes(args.output+"hof.sim", ['vertpos'], hof)

    print("unique evaluations", len(evaluator.evaluated_genos))
    print("total evaluations", sum(logbook.select('nevals')))
