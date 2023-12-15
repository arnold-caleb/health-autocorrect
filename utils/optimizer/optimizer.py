from hydra.utils import instantiate

def setup_optimizer_scheduler(cfg, model):
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    return optimizer, scheduler
