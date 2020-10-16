""" Parameters of the various scripts. """

RANDOM_SEED = 2345

# region Annotation task parameters
YEARS = [2006, 2007]
MAX_TUPLE_SIZE = 6
RANDOM = True
EXCLUDE_PILOT = True
ANNOTATION_TASK_SHORT_SIZE = 10000

LOAD_WIKI = True
WIKIPEDIA_FILE_NAME = "wikipedia_global"
CORRECT_WIKI = True
# endregion

# region Modeling task parameters
MIN_ASSIGNMENTS = 5
MIN_ANSWERS = 2
K_CROSS_VALIDATION = 5

VALID_PROPORTION = 0.25
TEST_PROPORTION = 0.25
RANKING_SIZE = 24
BATCH_SIZE = 4
CONTEXT_FORMAT = 'v0'
TARGETS_FORMAT = 'v0'
SYMBOLIC_FORMAT = "input"

# endregion

# region Models parameters
SCORES_NAMES = [
    'average_precision',
    'precision_at_10',
    'precision_at_9',
    'precision_at_8',
    'precision_at_7',
    'precision_at_6',
    'precision_at_5',
    'precision_at_4',
    'precision_at_3',
    'precision_at_2',
    'precision_at_1',
    'recall_at_10',
    'ndcg_at_10',
    'reciprocal_best_rank',
    'reciprocal_average_rank',
]

TASK_NAME = "context-dependent-same-type"
CONTEXT_MAX_SIZE = 750
SHOW_RANKINGS = 5
SHOW_CHOICES = 10

RANKER="generator"

BART_BEAM = 24
BART_LENPEN = 1.0
BART_MAX_LEN_B = 100
BART_MIN_LEN = 1
BART_NO_REPEAT_NGRAM_SIZE = 2
INFERENCE_BATCH_SIZE = 32
# endregion

# region Knowledge Graphs parameters
CATEGORY_MAX_DEPTH = 6
INFOBOX_MAX_DEPTH = 3
OUTPUT_CATEGORY_GRAPH = "category_graph.pkl"
OUTPUT_INFOBOX_GRAPH = "infobox_graph.pkl"
# endregion