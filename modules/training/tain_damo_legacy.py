from modules.training.damo_trainer import DamoTrainer
from modules.training.training_options import load_options_from_json
from modules.utils.paths import Paths


def train():
    json_path = Paths.support_data / "train_options.json"
    options = load_options_from_json(json_path)

    trainer = DamoTrainer(options)
    trainer.train()


if __name__ == '__main__':
    train()
