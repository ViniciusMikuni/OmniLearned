from omnilearned.train import run as train


def run(**config):
    return train(high_level=True, **config)
