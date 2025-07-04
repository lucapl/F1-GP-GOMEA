import argparse
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from dotenv import load_dotenv
from FramsticksLibCompetition import FramsticksLibCompetition

from src.gomea import eaGOMEA, forced_improvement, gom, override_nodes
from src.gpf1 import create_f1_pset, parse
from src.linkage import LinkageTreeFramsF1
from src.our_toolbox import OurToolbox
from src.utils.elitism import SaveBest
from src.utils.fpcontrol import fpenv_context_restore
from src.utils.stopping import EarlyStopper, earlyStoppingOrMaxIter

# from FramsticksLibCompetition import FramsticksLibCompetition
# SHOULD work properly after `uv sync`
# from framspy.src.FramsticksLibCompetition import FramsticksLibCompetition

load_dotenv()
# default values for --framslib and --sim_location
ENV_FRAMSTICKS_DLL = os.getenv("FRAMSTICKS_DLL_PATH", "./Framsticks52")
ENV_FRAMSPY_PATH = os.getenv("FRAMSPY_PATH", "./framspy")
# ENV_FRAMSPY_PATH = os.getenv("FRAMSPY_PATH", str(framspy.__path__[0]))


def prepare_gomea_parser(parser):
    parser.add_argument('-n', '--ngen', default=15, type=int, help="Number of generations to rund")
    parser.add_argument('-p', '--popsize', default=20, type=int, help="Size of population")
    parser.add_argument('-e', '--early_stop', default=10, type=int, help="Number of non-improving iterations till stopping")
    parser.add_argument('-g', '--initial_geno_mutations', default=100, type=int)
    parser.add_argument('--parts', nargs=2, type=int, default=[20, 30], help='Initial genotypes parts range')
    parser.add_argument('--neurons', nargs=2, type=int, default=[6, 8], help='Initial genotypes neurons range')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--MOCK_EVALS', action="store_true", help="RUNS random FUNCTION INSTEAD OF TRUE EVALUATION -> FOR TESTING ALGORITHMS")
    parser.add_argument('--sims',
                        nargs='+',
                        default=['eval-allcriteria.sim', 'f1_params.sim', 'recording-body-coords.sim'],
                        help='List of simulation files to use.')
    parser.add_argument('--no_forced_improv', action='store_true', help='Toggles forced improvement phase in GOMEA')
    parser.add_argument('--forced_improv_global', action='store_true', help="If true takes the best in history during forced improvement, else it takes the best from the population")
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
    parser.add_argument('-t', '--test_function', default=3, choices=[3, 4, 5], help="Which test function to evaluate")
    parser.add_argument('--criteria', default='COGpath', help="Name of the evaluation function criteria")

    return parser



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

    with fpenv_context_restore("loading Framsticks DLL"):
        framsLib = FramsticksLibCompetition(args.framslib, None, sim_formatted)
        framsLib.TEST_FUNCTION = args.test_function
        framsLib.SIMPLE_FITNESS_FORMAT = False

    # sometimes pandas crashes at print(df)...
    # print_fenv_state("Verify")
    # ✅ In pure Python: Use NumPy’s seterr
    # With NumPy, you can configure how it responds to FP events:
    #
    # Raise exceptions on specified FP errors
    np.seterr(all='raise')  # or specify individually, e.g. invalid='raise'
    # np.seterr(all='print')

    # After this, operations leading to nan, inf, divide-by-zero etc. will raise `FloatingPointError`
    # You can also wrap operations:
    # with np.errstate(divide='raise', invalid='raise'):
    #     x = np.log(-1)  # triggers FloatingPointError: invalid value encountered in log
    #     print("\n\n\n", flush=True)


    #####################
    # deap definitions
    #####################
    pset = create_f1_pset()
    # framsticks classes
    creator.create("FitnessMax", base.Fitness, weights=[1])
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = OurToolbox(args=args, framsLib=framsLib, pset=pset, if_forced_improv_save_best=args.forced_improv_global)

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
    #print(pop[0])

    ########################
    # MAIN ALGORITHM
    ########################
    try:
        new_pop, logbook, _linkage_log = eaGOMEA(
            pop,
            toolbox,
            start_gen=start_gen,
            logbook=logbook,
            stats=mstats,
            halloffame=hof,
            fmut=args.fmut,
            # checkpoint_freq=args.checkpoint_frequency,
            # checkpoint_name=checkpoints_out,
            verbose=args.verbose,
        )
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    if True:
        _cached_eval_ratio = toolbox.solution_cache_hits / (toolbox.solution_cache_hits + toolbox.solution_cache_misses)
        print(f"Eval. cache stats: {_cached_eval_ratio:.1%}  "
            f"{toolbox.solution_cache_hits:5} reused"
            f" and {toolbox.solution_cache_misses} simulated."
            )

    # from rich import print as rprint
    # rprint("\n\nLogbook: ", logbook)  # [{'gen': 0, 'nevals': 123}, ...]
    # # defaultdict with 'fitness', 'genotype_length'
    # rprint("\n\nLogbook.chapters: ", logbook.chapters)

    # log_df = pd.json_normalize({**dict(logbook), **dict(logbook.chapters)})
    log_df = deap_log_with_multi_stats_to_df(logbook)
    # print("\n\nas data frame: ", log_df.shape)
    # print(log_df.dtypes)
    print("\n\n")
    print(log_df.describe())
    print()

    # parsed_args.out_prefix + "_stats.csv"
    out_log = "test" + "_stats.csv"
    log_df.to_csv(out_log, sep=";", index=False)
    print("Saved", out_log)

    #######################
    # saving outputs
    #######################
    framsLib.end()


def deap_log_with_multi_stats_to_df(logbook: tools.Logbook) -> pd.DataFrame:
    records = []
    # log_df = pd.DataFrame.from_records(logbook)  # without nesting
    # log_df = pd.json_normalize(logbook)

    # from rich import print as rprint

    for each_row in logbook:
        # gen, nevals
        records.append(each_row)
    # rprint("records: ", records)

    sub_dfs = []
    for chapter_name, ls_subrecords in logbook.chapters.items():
        chapter_df = pd.DataFrame.from_records(ls_subrecords)
        rename_dict = {
            col: f"{chapter_name}_{col}"
            for col in chapter_df.columns if col not in ("gen", "nevals")
            #  if col != "gen"
        }
        chapter_df = chapter_df.drop(columns=["nevals"])
        chapter_df = chapter_df.rename(columns=rename_dict)
        # rprint(f"\n{chapter_name} df:\n", chapter_df)
        # rprint("...records were: ", ls_subrecords)
        sub_dfs.append(chapter_df.set_index('gen'))

    log_df = pd.DataFrame.from_records(records).set_index('gen')
    log_df = log_df.join(sub_dfs, validate="one_to_one")
    # result_df = log_df.merge(df, on='gen', how='outer')
    # rprint("\n\n\njoined\n", log_df)
    return log_df


if __name__ == "__main__":
    main()
