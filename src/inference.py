import argparse
import datetime
import json
import logging
import os.path

import numpy as np
import torch

from config.config import CFG
from utils.config import Config


def inference(folder, inference_data) -> None:
    """
    Evaluate model
    """
    config = Config.from_json(CFG)
    folder = os.path.join(config.root_dir, 'saved', folder)
    logging.basicConfig(filename=os.path.join(folder, 'inference.log'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info('config = \n{}'.format(json.dumps(CFG, indent=4)))
    logger.info('===> Inference started.')

    # fix random seeds for reproducibility
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(config.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = torch.load(os.path.join(folder, 'best_model.pth'))
    model.to(device)
    logger.info(model)
    logger.info('Length of inference sequence: {}'.format(len(inference_data)))

    model.eval()
    with torch.no_grad():
        x = torch.tensor(inference_data, dtype=torch.float).to(device)
        output = model(x)
        prediction = torch.argmax(output, dim=1)
        result = prediction.tolist()

    with open(os.path.join(folder, 'inference_result.txt'), 'w') as f:
        f.writelines('index, result\n')
        for i, line in enumerate(result):
            f.writelines(f'{i}, {line}\n')

    logger.info('===> Inference finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default=None, type=str, required=True,
                        help='name of folder which contains \'best_model.pth\' under '
                             'the folder \'saved\' (eg: \'2023-01-01-00-00-00\')')
    args = parser.parse_args()

    # Random data for inference example
    data = np.random.rand(10, 1024, 7)

    # Inference
    inference(folder=args.folder, inference_data=data)
