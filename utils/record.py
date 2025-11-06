import logging
from torch.utils.tensorboard import SummaryWriter



def log_info(type:str, name:str, info, step=None, record_tool='wandb'):
    '''
    type: the info type mainly include: image, scalar (tensorboard may include hist, scalars)
    name: replace the info name displayed in wandb or tensorboard
    info: info to record
    '''
    if record_tool=='wandb':
        import wandb
    elif record_tool=='swanlab':
        import swanlab
    if type == 'image':
        if record_tool == 'tensorboard':
            writer.add_image(name, info, step)
        if record_tool == 'wandb':
            wandb.log({name: wandb.Image(info)})
        if record_tool == 'swanlab':
            swanlab.log({name: swanlab.Image(info)})

    if type == 'scalar':
        if record_tool == 'tensorboard':
            writer.add_scalar(name, info, step)
        if record_tool == 'wandb':
            wandb.log({name:info})
        if record_tool == 'swanlab':
            swanlab.log({name:info})
    if type == 'histogram':
        writer.add_histogram(name, info, step)

def logging_config(args, symbol):
    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()
    console = logging.StreamHandler()
    if args.log_level == 'INFO':
        console.setLevel(logging.INFO)
    elif args.log_level == 'DEBUG':
        console.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    formatter = logging.Formatter(str(symbol) + 
        ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)
    # logging.getLogger().info("test")
    logging.basicConfig()
    logger = logging.getLogger()
    if args.log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif args.log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    logging.info(args)

def record(record_tool, **kwargs):
    if 'scalar' in kwargs:
        for keys in kwargs['scalar']:
            log_info('scalar', keys ,kwargs['scalar'][keys], kwargs['step'], record_tool)
    
    if 'image' in kwargs:
        for keys in kwargs['image']:
            log_info('image', keys ,kwargs['image'][keys], kwargs['step'], record_tool)