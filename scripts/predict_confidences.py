import click
import yaml
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F

import mos4d.datasets.datasets as datasets
import mos4d.models.models as models


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default="./config/gene_test_config.yaml",
)
@click.option(
    "--sequences",
    "-seq",
    type=int,
    help="run inference on a specific sequence, otherwise, default test split is used",
    default=None,
    multiple=True,
)
def main(config, sequences):
    # config
    cfg = yaml.safe_load(open(config))  # cfg = torch.load(weights)["hyper_parameters"]'
    assert cfg["EXPT"]["MODE"] == "TEST"
    num_device = cfg["TEST"]["NUM_DEVICES"]
    ckpt_path = cfg["TEST"]["CKPT"]
    # dataset
    dataset = cfg["TEST"]["DATASET"]
    cfg["DATASET"][dataset]["TRAIN"] = cfg["DATASET"][dataset]["TEST"]  # in test mode
    cfg["DATASET"][dataset]["VAL"] = cfg["DATASET"][dataset]["TEST"]  # in test mode
    if sequences:
        cfg["DATASET"][dataset]["TEST"] = list(sequences)
    # method params
    strategy = cfg["TEST"]["STRATEGY"]
    bayes_prior = cfg["TEST"]["BAYES_PRIOR"]
    delta_t = cfg["TEST"]["DELTA_T"]  # cfg["MODEL"]["DELTA_T_PREDICTION"]
    transform = cfg["DATA"]["TRANSFORM"]

    # Load data and model for different datasets
    data = []
    data = datasets.KittiSequentialModule(cfg)
    data.setup()

    # Load ckeckpoint model
    ckpt = torch.load(ckpt_path)
    model = models.MOSNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()
    # Setup trainer
    trainer = Trainer(accelerator="gpu", devices=num_device, logger=False)
    # Inference
    trainer.predict(model, data.test_dataloader())  # predict confidence scores (softmax output)


if __name__ == "__main__":
    main()
