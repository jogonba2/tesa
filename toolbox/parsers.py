from toolbox.parameters import *
from toolbox.paths import *

from argparse import ArgumentParser


def standard_parser():
    """ Create an argument parser, add the parsing arguments relative to the options and return it. """

    ap = ArgumentParser()

    ap.add_argument("--random_seed",
                    type=int, default=RANDOM_SEED,
                    help="Random seed for numpy and torch.")
    ap.add_argument("--root",
                    type=str, default="",
                    help="Path to the root of the project.")
    ap.add_argument("--results_path",
                    type=str, default=None,
                    help="Path to the result folder.")
    ap.add_argument("--no_save",
                    action='store_true',
                    help="No save option.")
    ap.add_argument("--silent",
                    action='store_true',
                    help="Silence option.")

    return ap

def add_knowledge_graphs_arguments(ap):
    """
    Add to the argument parser the parsing arguments relative to the annotations.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the knowledge graphs arguments.
    """

    ap.add_argument("-dc", "--max_depth_category",
                    type=int, default=CATEGORY_MAX_DEPTH,
                    help="Maximum depth for the category traversal on entities of TESA")
    ap.add_argument("-di", "--max_depth_infobox",
                    type=int, default=INFOBOX_MAX_DEPTH,
                    help="Maximum depth for the infobox traversal on entities of TESA")
    ap.add_argument("--annotations_path",
                    type=str, default=ANNOTATION_TASK_RESULTS_PATH,
                    help="Path to the annotation task results folder.")


def add_annotations_arguments(ap):
    """
    Add to the argument parser the parsing arguments relative to the annotations.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the arguments.
    """

    ap.add_argument("--min_assignments",
                    type=int, default=MIN_ASSIGNMENTS,
                    help="Minimum number of assignments an annotator has to have done not be filtered out.")
    ap.add_argument("--min_answers",
                    type=int, default=MIN_ANSWERS,
                    help="Minimum number of actual answers an annotation task has to have not be filtered out.")
    ap.add_argument("--exclude_pilot",
                    type=bool, default=EXCLUDE_PILOT,
                    help="Whether or not to exclude the pilot.")
    ap.add_argument("--annotations_path",
                    type=str, default=ANNOTATION_TASK_RESULTS_PATH,
                    help="Path to the annotation task results folder.")


def add_task_arguments(ap):
    """
    Add to the argument parser the parsing arguments relative to the modeling task.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the arguments.
    """

    ap.add_argument("-t", "--task",
                    type=str, default=TASK_NAME,
                    help="Name of the modeling task version.")
    ap.add_argument("-vp", "--valid_proportion",
                    type=float, default=VALID_PROPORTION,
                    help="Proportion of the validation set.")
    ap.add_argument("-tp", "--test_proportion",
                    type=float, default=TEST_PROPORTION,
                    help="Proportion of the test set.")
    ap.add_argument("-rs", "--ranking_size",
                    type=int, default=RANKING_SIZE,
                    help="Size of the ranking tasks.")
    ap.add_argument("-bs", "--batch_size",
                    type=int, default=BATCH_SIZE,
                    help="Size of the batches of the task.")
    ap.add_argument("-cf", "--context_format",
                    type=str, default=CONTEXT_FORMAT,
                    help="Version of the context format.")
    ap.add_argument("-tf", "--targets_format",
                    type=str, default=TARGETS_FORMAT,
                    help="Version of the targets format.")
    ap.add_argument("--context_max_size",
                    type=int, default=CONTEXT_MAX_SIZE,
                    help="Maximum number of tokens in the context.")
    ap.add_argument("--task_path",
                    type=str, default=MODELING_TASK_RESULTS_PATH,
                    help="Path to the task folder.")
    ap.add_argument("--cross_validation",
                    action='store_true',
                    help="Cross validation option.")
    ap.add_argument("--k_cross_validation",
                    type=int, default=K_CROSS_VALIDATION,
                    help="Number of folds for cross validation.")
    ap.add_argument("--generation",
                    action='store_true',
                    help="Generation finetuning option.")
    ap.add_argument("--classification",
                    action='store_true',
                    help="Classification finetuning option.")
    ap.add_argument("--finetuning_data_path",
                    type=str, default=FINETUNING_DATA_PATH,
                    help="Path to the finetuning data folder.")
    ap.add_argument("--graphs_path",
                    type=str, default=KNOWLEDGE_GRAPHS_RESULTS_PATH,
                    help="Path to the results folder for knowledge graphs.")
    ap.add_argument("--category_graph_file",
                    type=str, default=None,
                    help="Path for the category graph.")
    ap.add_argument("--infobox_graph_file",
                    type=str, default=None,
                    help="Path for the category graph.")
    ap.add_argument("--category_symbolic_algo",
                    type=str, default=None,
                    help="Method for extracting symbolic information from the category graph")
    ap.add_argument("--infobox_symbolic_algo",
                    type=str, default=None,
                    help="Method for extracting symbolic information from the infobox graph")
    ap.add_argument("--symbolic_algo",
                    type=str, default=None,
                    help="String with the symbolic algorithms joined by '-'"
                         "(wrapper for simplify run_models with symbolic information)")
    ap.add_argument("--symbolic_format",
                    type=str, default=SYMBOLIC_FORMAT,
                    help="Determines where the symbolic information is used (as inputs or as candidates)")
    ap.add_argument("--soft_labels",
                    action='store_true',
                    help="Distinguish labels of graph information and candidates ('partial_aggregation')")


def add_model_arguments(ap):
    """
    Add to the argument parser the parsing arguments relative to the model.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the arguments.
    """

    ap.add_argument("-m", "--model",
                    type=str, required=False,
                    help="Name of the model.")
    ap.add_argument("--scores_names",
                    type=list, default=SCORES_NAMES,
                    help="Names of the scores to use.")
    ap.add_argument("--tensorboard_logs_path",
                    type=str, default=TENSORBOARD_LOGS_PATH,
                    help="Path of the tensorboard logs folder.")
    ap.add_argument("--pretrained_path",
                    type=str, default=PRETRAINED_MODELS_PATH,
                    help="Path to the pretrained model folder.")
    ap.add_argument("--checkpoint_file",
                    type=str, default=None,
                    help="Name of BART's checkpoint file.")
    ap.add_argument("--checkpoint_generative",
                    type=str, default=None,
                    help="Name of generative BART's checkpoint file. (Tesa trained).")
    ap.add_argument("--checkpoint_discriminative",
                    type=str, default=None,
                    help="Name of discriminative BART's checkpoint file (TESA trained).")
    ap.add_argument("--ranker",
                    type=str, default=None, choices={"generative", "discriminative"},
                    help="Model used for ranking candidates in the wild.")
    ap.add_argument("--model_random_seed",
                    type=int, default=RANDOM_SEED,
                    help="Random seed of the model.")
    ap.add_argument("--show_rankings",
                    type=int, default=SHOW_RANKINGS,
                    help="Number of rankings to show.")
    ap.add_argument("--show_choices",
                    type=int, default=SHOW_CHOICES,
                    help="Number of choices to show per ranking.")
    ap.add_argument("--error_analysis",
                    action="store_true",
                    help="Option to save a json file with information for error analysis.")
    ap.add_argument("--bart_beam",
                    type=int, default=BART_BEAM,
                    help="Parameter beam for generator BART.")
    ap.add_argument("--bart_lenpen",
                    type=float, default=BART_LENPEN,
                    help="Parameter lenpen for generator BART.")
    ap.add_argument("--bart_max_len_b",
                    type=int, default=BART_MAX_LEN_B,
                    help="Parameter max_len_b for generator BART.")
    ap.add_argument("--bart_min_len",
                    type=int, default=BART_MIN_LEN,
                    help="Parameter bart_min_len for generator BART.")
    ap.add_argument("--bart_no_repeat_ngram_size",
                    type=int, default=BART_NO_REPEAT_NGRAM_SIZE,
                    help="Parameter no_repeat_ngram_size for generator BART.")
    ap.add_argument("--word2vec",
                    action="store_true",
                    help="Load Word2Vec embedding.")
    ap.add_argument("--bart",
                    action="store_true",
                    help="Load a BART model.")
    ap.add_argument("--tensorboard",
                    action="store_true",
                    help="Option to use tensorboard.")
    ap.add_argument("--show",
                    action="store_true",
                    help="Option to show some examples instead of ranking.")
    ap.add_argument("--random_examples",
                    action="store_true",
                    help="Option to show random examples (if show).")
    ap.add_argument("--custom_examples",
                    action="store_true",
                    help="Option to show the custom examples (if show).")
    ap.add_argument("--unseen_examples",
                    action="store_true",
                    help="Option to show random unseen examples (if show).")
    ap.add_argument("-bst", "--inference_batch_size",
                    type=int, default=INFERENCE_BATCH_SIZE,
                    help="Batch size for inference")
    ap.add_argument("--wild", default=False,
                    action="store_true",
                    help="Option for wild evaluation (candidates extracted from the generative model)")
    ap.add_argument("--rerank", default=False,
                    action="store_true",
                    help="Option for wild evaluation (reranking with the ranker or not)")
    ap.add_argument("--validation", default=True,
                    action="store_true",
                    help="Flag to evaluate on the validation set")