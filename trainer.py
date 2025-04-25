    def _train_epoch2(self, model1, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss2
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses, a, seq_output, logits  = model1.calculate_loss(interaction)
                a = a.detach()
                seq_output = seq_output.detach()
                logits = logits.detach()
                losses2 = loss_func(interaction, seq_output, a, logits)

            if isinstance(losses2, tuple):
                loss = sum(losses2)
                loss_tuple = tuple(per_loss.item() for per_loss in losses2)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses2
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses2.item()
                )
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss
        
    def fit2(
        self,
        model1,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch2(
                model1, train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
