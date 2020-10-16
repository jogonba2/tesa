"""
Script to run a Model on a task.

Usages:
    python run_model.py -m random
    python run_model.py -m frequency --show
    python run_model.py -m summaries_average_embedding --word2vec
"""

from toolbox.parsers import standard_parser, add_task_arguments, add_model_arguments
from toolbox.utils import (load_task, get_pretrained_model, to_class_name,
                           load_task_absolute_path, get_bart)
import modeling.models as models


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_task_arguments(ap)
    add_model_arguments(ap)

    return ap.parse_args()

def static_ranking(args):
    task = load_task(args)
    pretrained_model = get_pretrained_model(args)
    model = getattr(models, to_class_name(args.model))(args=args, pretrained_model=pretrained_model)
    model.play(task=task, args=args)

def wild_ranking(args):

    task = load_task_absolute_path(args.task_path)
    discriminative_model = None


    if args.ranker == "discriminative":
        discriminative_model = get_bart(args.pretrained_path,
                                        args.checkpoint_discriminative)

    generative_model = get_bart(args.pretrained_path,
                                args.checkpoint_generative)

    model = models.WildBart(args, generative_model,
                            discriminative_model=discriminative_model)

    model.wild_play(task, args)

def main():
    """ Makes a model run on a task. """

    args = parse_arguments()

    if args.wild:
        wild_ranking(args)
    else:
        static_ranking(args)


if __name__ == '__main__':
    main()
