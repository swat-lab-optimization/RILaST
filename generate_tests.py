import importlib
import logging
import os
import shutil
from datetime import datetime
from typing import Type, Optional

from rilast.common.get_convergence import get_convergence
from rilast.common.get_stats import get_stats
from rilast.common.get_test_suite import get_test_suite
from rilast.common.save_tc_results import save_tc_results, create_summary
from rilast.executors.beam_executor import BeamExecutor
from rilast.executors.obstacle_scene_executor import ObstacleSceneExecutor
from rilast.common.parse_arguments import parse_arguments_test_generation

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").disabled = True
log = logging.getLogger(__name__)

def setup_logging(debug: bool = False, log_to: Optional[str] = "log.txt") -> None:
    """
    Set up the logging system.
    """
    term_handler = logging.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to:
        file_handler = logging.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += f", writing logs to file: {log_to}"

    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )

    logging.info(start_msg)

def generate_tests(
    runs: int = 1,
    generator_class: Type = None,
    alg: str = "ga",
    cross: str = "sbx",
    mut: str = "pm",
    add_info: str = ""
) -> None:
    """
    Run the optimization process.
    """
    log.info("Starting optimization")
    log.info(f"Number of runs: {runs}")
    log.info(f"Generator: {generator_class}")
    log.info(f"Algorithm: {alg}")
    log.info(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log.info("----------------------------------------------------")

    tc_stats, tcs, tcs_convergence, all_tests = {}, {}, {}, {}
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")
    #exp_id = now.strftime("%d-%m-%Y-%H-%M-%S")
    root_path = os.path.join("experiments")
    log.info(f"Saving results to: {root_path}")

    for run in range(runs):
        log.info(f"Run {run}")

        sim_save_path = setup_simulation_path(generator_class, dt_string, alg, cross, mut, add_info, root_path, run)
        generator = generator_class(save_path=sim_save_path)

        try:
            res, test_executor = generator.start(alg, cross, mut)
        except Exception as e:
            log.error(f"Error while running generator: {e}")
            res, test_executor = generator.res, generator.test_executor

        log.info(f"Run {run} finished")
        log.info("----------------------------------------------------")

        tc_stats[f"run{run}"] = get_stats(res)
        tcs_convergence[f"run{run}"] = get_convergence(res)
        tcs[f"run{run}"] = get_test_suite(res)
        all_tests[f"run{run}"] = test_executor.test_dict

        save_tc_results(
            dt_string,
            tc_stats,
            tcs,
            tcs_convergence,
            all_tests,
            "stats",
            alg,
            cross,
            f"{mut}_{add_info}",
            root_path=root_path,
        )

        handle_executor_results(test_executor, dt_string, alg, cross, mut, run)

def setup_simulation_path(generator_class, dt_string, alg, cross, mut, add_info, root_path, run):
    if generator_class.__name__ in ["LKASTestGenerator", "LatentLKASTestGenerator"]:
        sim_save_path_base = os.path.join(
            root_path,
            f"{dt_string}_stats_BEAM_NG_{alg}_{cross}_{mut}{add_info}"
        )
        sim_save_path = os.path.join(sim_save_path_base, str(run))
        os.makedirs(sim_save_path, exist_ok=True)
        return sim_save_path
    return None

def handle_executor_results(test_executor, dt_string, alg, cross, mut, run):
    if isinstance(test_executor, BeamExecutor):
        create_summary(test_executor.save_path, test_executor.beamng_executor.get_stats())
        test_executor.beamng_executor.close()
        del test_executor.beamng_executor

    if isinstance(test_executor, ObstacleSceneExecutor):
        save_uav_test_cases(test_executor, dt_string, alg, cross, mut, run)

def save_uav_test_cases(test_executor, dt_string, alg, cross, mut, run):
    uav_all_tests = test_executor.uav_test_dict
    test_cases = [uav_all_tests[tc]["test"] for tc in uav_all_tests if uav_all_tests[tc]["info"] == "simulation"]

    tests_folder = os.path.join(f"{dt_string}_UAV_executor_{alg}_{cross}_{mut}", f"run_{run}")
    os.makedirs(tests_folder, exist_ok=True)

    for i, test_case in enumerate(test_cases):
        test_case.save_yaml(os.path.join(tests_folder, f"test_{i}.yaml"))
        shutil.copy2(test_case.log_file, os.path.join(tests_folder, f"test_{i}.ulg"))
        shutil.copy2(test_case.plot_file, os.path.join(tests_folder, f"test_{i}.png"))

    log.info(f"{len(test_cases)} test cases generated")
    log.info(f"Output folder: {tests_folder}")

if __name__ == "__main__":
    args = parse_arguments_test_generation()
    module_name = args.module_name
    class_name = args.class_name
    runs = args.runs
    alg, cross, mut, add_info = args.algorithm, args.crossover, args.mutation, args.add_info

    setup_logging(debug=True, log_to="log.txt")
    module = importlib.import_module(module_name)
    the_class = getattr(module, class_name)

    generate_tests(runs=runs, generator_class=the_class, alg=alg, cross=cross, mut=mut, add_info=add_info)
