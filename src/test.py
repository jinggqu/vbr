import argparse
import datetime
import logging
import os.path

import numpy as np
import torch
from sklearn import metrics

from config.config import CFG
from dataloader.custom_dataset import CustomDataset
from dataloader.dataloader import DataLoader
from utils.config import Config
from utils.util import plot_matrix, plot_auc


def test(folder) -> None:
    """
    Evaluate model
    """
    config = Config.from_json(CFG)
    folder = os.path.join(config.root_dir, 'saved', folder)
    logging.basicConfig(filename=os.path.join(folder, 'test.log'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('===> Testing started.')
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(CFG)

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

    data_loader = DataLoader(config)
    test_data = data_loader.load_data()['test']
    logger.info(f'Length of test_data: {len(test_data)}')
    test_dataloader = torch.utils.data.DataLoader(
        CustomDataset(config, test_data),
    )

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            x, y = batch[0].to(device), batch[1].to(device)
            y_true.append(y.item())
            output = model(x)
            prediction = torch.argmax(output, dim=1)
            y_pred.append(prediction.item())

    with open(os.path.join(folder, 'test_result.txt'), 'w') as f:
        f.writelines('\n'.join(str(line) for line in y_pred))

    classes = data_loader.label_classes()
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=classes.argsort())

    np.savetxt(os.path.join(folder, 'confusion_matrix.txt'), matrix, fmt='%d', delimiter=',')
    plot_matrix(matrix, classes, x_label='Predicted Label', y_label='True Label',
                save_to=os.path.join(folder, 'confusion_matrix.png'), ticks_rotation=0, show=False)

    average = 'micro'
    test_metrics = ['accuracy:\t' + str(metrics.accuracy_score(y_true, y_pred)),
                    'precision:\t' + str(metrics.precision_score(y_true, y_pred, average=average)),
                    'recall: \t' + str(metrics.recall_score(y_true, y_pred, average=average)),
                    'f1_score:\t' + str(metrics.f1_score(y_true, y_pred, average=average))]

    with open(os.path.join(folder, 'test_metrics.txt'), 'w') as f:
        f.writelines('\n'.join(test_metrics))

    # fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    # plot_auc(fpr=fpr, tpr=tpr, save_to=os.path.join(folder, 'auc.pdf'), ticks_rotation=0, show=False)

    logger.info('===> Testing finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default=None, type=str, required=True,
                        help='name of folder which contains \'best_model.pth\' under '
                             'the folder \'saved\' (eg: \'2023-01-01-00-00\')')
    args = parser.parse_args()
    test(folder=args.folder)
