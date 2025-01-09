
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from typing import Text, Dict
import yaml
import numpy as np
from src.utils.utils import load_model

from src.report.visualization import plot_confusion_matrix
from src.utils.logs import get_logger


def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('EVALUATE',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    logger.info('Load model')
    model_path = config['train']['model_path']
    model = load_model(model_path)

    logger.info('Load test dataset')
    test_indices = np.load(config['data_split']['train_test_split_path'])['test_indices']
    data = np.load(config['featurize']['features_path'])
    X_test = data['subspace_embeddings'][test_indices]
    y_test = data['labels'][test_indices]

    logger.info('Evaluate (build report)')
    prediction = model.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=prediction)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')
    cm = confusion_matrix(prediction, y_test)

    report = {
        'f1': f1,
        'acc': acc,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }
    print(f"Accuracy: {acc}, \t F1 score: {f1}")

    logger.info('Save metrics')
    # save f1 metrics file
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']

    json.dump(
        obj={'accuracy': report['acc'], 'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )

    logger.info(f'F1 metrics file saved to : {metrics_path}')

    logger.info('Save confusion matrix')
    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=config['evaluate']['target_names'],
                                normalize=False)
    confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path)
    logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')


if __name__ == '__main__':
    evaluate_model('../../params.yaml')
#     args_parser = argparse.ArgumentParser()
#     args_parser.add_argument('--config', dest='config', required=True)
#     args = args_parser.parse_args()
#
#     evaluate_model(config_path=args.config)