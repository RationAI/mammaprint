# Copyright (c) The RationAI team.

from functools import partial

from mammaprint.cli_helper import main, preset_hydra_main_decorator


# this file is there so that users do not have to specify stage in the config file
if __name__ == "__main__":
    train_main = partial(
        main, stage="fit"
    )  # partial is used to pass the stage argument to main
    hydra_train_main = preset_hydra_main_decorator(train_main)
    hydra_train_main()
