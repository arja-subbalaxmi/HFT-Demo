import logging, colorlog
def get_logger(name):
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s.%(msecs)03d | %(name)-8s | %(message)s",
        datefmt="%H:%M:%S", reset=True, log_colors={'INFO':'green','WARNING':'yellow'}))
    log = logging.getLogger(name)
    log.setLevel(logging.INFO); log.addHandler(handler)
    return log
