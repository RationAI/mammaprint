from functools import partial

from histopipe.cli_helper import main, preset_hydra_main_decorator


if __name__ == "__main__":
    train_main = partial(
        main, stage="test"
    )  # partial is used to pass the stage argument to main
    hydra_train_main = preset_hydra_main_decorator(train_main)
    hydra_train_main()
