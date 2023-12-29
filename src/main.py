"""
    Main
"""


import json
import logging as lg
import os
from datetime import datetime
from sys import argv, stdout
from time import time

import spacy
from torch import multiprocessing as tmp

from ainet_executor import AiNetExecutor

stdout.reconfigure(line_buffering=True)  # type: ignore


if __name__ == "__main__":
    spacy.prefer_gpu()  # type: ignore
    script_start = time()
    tmp.set_start_method("spawn", force=True)

    root_log = lg.getLogger("root")

    file_logger = lg.FileHandler(
        filename=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs",
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        ),
        encoding="utf-8",
    )

    file_logger.setLevel(lg.DEBUG)

    file_logger.setFormatter(
        lg.Formatter(
            "(%(asctime)s)[%(levelname)s:%(name)s] %(module)s.%(filename)s.%(funcName)s => | %(message)s |"  # noqa: E501
        )
        # JSON LOG
        # "{\"execution_time\":\"%(asctime)s\",\"log_level\":\"%(levelname)s:%(name)s\",\"method\":\"%(module)s.%(filename)s.%(funcName)s\",\"message\":\"%(message)s\"}")  # noqa: E501
    )
    file_logger.addFilter(lg.Filter(name="root"))

    stdout_logger = lg.StreamHandler()
    stdout_logger.setLevel(lg.ERROR)

    root_log.setLevel(lg.DEBUG)
    root_log.addHandler(file_logger)
    root_log.addHandler(stdout_logger)

    results: list[list[float | str]] = []

    lg.log(lg.DEBUG, "script started")
    try:
        EXECUTION_PLAN_FILE = "executions.json"

        if len(argv) > 1:
            ARGV_PLAN_FILE = argv[1]

            assert os.path.exists(ARGV_PLAN_FILE), f"file '{ARGV_PLAN_FILE}' does not exist"

            EXECUTION_PLAN_FILE = argv[1]

        lg.log(lg.INFO, "reading execution plan file '%s'", EXECUTION_PLAN_FILE)

        executions: dict = {}

        start = time()

        with open(EXECUTION_PLAN_FILE, "r", encoding="utf-8") as json_file:
            executions = json.load(json_file)
            lg.log(lg.DEBUG, "reading execution plan took %0.5fs ", time() - start)

            lg.log(lg.INFO, "execution plan '%s' loaded", EXECUTION_PLAN_FILE)

        executor = AiNetExecutor(logger=root_log)
        executor.execute(executions_plans=executions["executions_plans"])
    except Exception as ex:
        root_log.error(ex, exc_info=True)
