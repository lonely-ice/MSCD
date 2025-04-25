def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):

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
