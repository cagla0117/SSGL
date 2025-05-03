from reckit import Configurator
from importlib.util import find_spec
from importlib import import_module
from reckit import typeassert
import os
import sys
import numpy as np
import random
import torch

def _set_random_seed(seed=2020):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")


@typeassert(recommender=str)
def find_recommender(recommender):
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None

    for tdir in model_dirs:
        spec_path = ".".join(["model", tdir, recommender])
        if find_spec(spec_path):
            module = import_module(spec_path)
            break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender

if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_dir = os.path.abspath(os.path.dirname(__file__))  # Geçerli dizini otomatik alır
        data_dir = os.path.join(root_dir, 'dataset')
    else:
        root_dir = os.path.abspath(os.path.dirname(__file__))  # Geçerli dizini otomatik alır
        data_dir = os.path.join(root_dir, 'dataset')

    experiment_settings = [
        {"name": "long_tail", "long_tail": True, "short_tail": False, "cluster_pruning": False},
        {"name": "short_tail", "long_tail": False, "short_tail": True, "cluster_pruning": False},
        {"name": "cluster_pruning", "long_tail": False, "short_tail": False, "cluster_pruning": True},
    ]

    results = []
    metric_names = []

    for setting in experiment_settings:
        print(f"\n🔁 Starting experiment: {setting['name']}")
        config = Configurator(root_dir, data_dir)
        config.add_config(os.path.join(root_dir, "NeuRec.ini"), section="NeuRec")
        config.parse_cmd()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])
        _set_random_seed(config["seed"])
        Recommender = find_recommender(config.recommender)

        model_cfg = os.path.join(root_dir, "conf", config.recommender + ".ini")
        config.add_config(model_cfg, section="hyperparameters", used_as_summary=True)

        recommender = Recommender(config)

        # Patch SGL.create_adj_mat to inject correct pruning flags
        original_create_adj_mat = recommender.create_adj_mat
        def patched_create_adj_mat(*args, **kwargs):
            kwargs.update({
                "long_tail": setting["long_tail"],
                "short_tail": setting["short_tail"],
                "cluster_pruning": setting["cluster_pruning"]
            })
            return original_create_adj_mat(*args, **kwargs)
        recommender.create_adj_mat = patched_create_adj_mat

        # Train and evaluate
        recommender.train_model()
        recommender.lightgcn.eval()
        result, _ = recommender.evaluator.evaluate(recommender)
        results.append({
            "experiment": setting["name"],
            "metrics": result.tolist()
        })

        # metric adlarını sadece bir kere çekiyoruz
        if not metric_names:
            raw_str = recommender.evaluator.metrics_info()
            metrics_part = raw_str.split("metrics:\t")[1]
            metric_names = metrics_part.strip().split("\t")

    # 📊 EXPERIMENT TABLOSU
    print("\n📊 EXPERIMENT RESULTS TABLE")
    print("Experiment\t" + "\t".join(metric_names))
    for entry in results:
        values = ["%.4f" % v for v in entry["metrics"]]
        print(entry["experiment"] + "\t" + "\t".join(values))
