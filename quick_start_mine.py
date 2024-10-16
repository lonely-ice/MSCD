def run(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp
        
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            runs,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):

    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger1 = getLogger()
    logger1.info(sys.argv)
    logger1.info(config)
    logger2 = getLogger()
    logger2.info(sys.argv)
    logger2.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger1.info(dataset)
    logger2.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model1 = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger1.info(model1)
    model2 = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger2.info(model2)

    transform1 = construct_transform(config)
    flops1 = get_flops(model1, dataset, config["device"], logger1, transform1)
    logger1.info(set_color("FLOPs", "blue") + f": flops1")
    
    transform2 = construct_transform(config)
    flops2 = get_flops(model2, dataset, config["device"], logger2, transform2)
    logger2.info(set_color("FLOPs", "blue") + f": flops2")

    # trainer loading and initialization
    trainer1 = get_trainer(config["MODEL_TYPE"], config["model"])(config, model1)
    trainer2 = get_trainer(config["MODEL_TYPE"], config["model"])(config, model2)

    # model training
    best_valid_score1, best_valid_result1 = trainer1.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )
    print("#####################")
    best_valid_score2, best_valid_result2 = trainer2.fit2(
        model1, train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )
    
    # model evaluation
    test_result1 = trainer1.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )
    test_result2 = trainer1.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger1.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger1.info(set_color("best valid ", "yellow") + f": {best_valid_result1}")
    logger1.info(set_color("test result", "yellow") + f": {test_result1}")

    result1 = {
        "best_valid_score1": best_valid_score1,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result1": best_valid_result1,
        "test_result1": test_result1,
    }
    
    logger2.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger2.info(set_color("best valid ", "yellow") + f": {best_valid_result2}")
    logger2.info(set_color("test result", "yellow") + f": {test_result2}")

    result2 = {
        "best_valid_score2": best_valid_score2,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result2": best_valid_result2,
        "test_result2": test_result2,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result1, result2  # for the single process


def runs(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of runs should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run(
        *args[:3],
        **kwargs,
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
