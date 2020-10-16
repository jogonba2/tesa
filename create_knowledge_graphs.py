"""
Script to create the knowledge graphs for TESA task

Usage (scratch): python create_knowledge_graphs.py --results ./results/knowledge_graphs/ \
                                                   --max_depth_category 6 \
                                                   --max_depth_infobox 3

Usage: python create_knowledge_graphs.py --results ./results/knowledge_graphs/

"""

from modeling.knowledge_graphs import (
    CategoryDatabase, CategoryGraph,
    InfoboxGraph, FullGraph
)

from toolbox.parsers import standard_parser, add_knowledge_graphs_arguments
from pywikibot import error as log_error
from pywikibot import Error as BotError
import pickle as pkl


def parse_arguments():
    """ Use arparse to parse the input arguments and
        return it as a argparse.ArgumentParser.
    """

    ap = standard_parser()
    add_knowledge_graphs_arguments(ap)
    return ap.parse_args()


def build_infobox_graph(args, entities):
    info_graph = InfoboxGraph(filename=args.results_path +
                                       "infobox_graph_depth-%d.pkl" % (args.max_depth_infobox),
                              cachename=args.results_path +
                                        "infobox_dump.pkl")

    info_graph.build_graph(entities, args.max_depth_infobox)
    return info_graph


def build_category_graph(args, entities):

    cat_db = CategoryDatabase(rebuild=False,
                              filename=args.results_path + "category.dump.bz2")
    cat_graph = CategoryGraph(cat_db,
                              filename=args.results_path +
                                       "category_graph_depth-%d.pkl" % (args.max_depth_category))

    try:
        cat_graph.build_graph(entities, args.max_depth_category)
    except BotError:
        log_error("Fatal pywikibot error:", exc_info=True)
    finally:
        cat_db.dump()

    return cat_graph


def build_full_graph(args, infobox_graph, category_graph):

    full_graph = FullGraph(filename=args.results_path +
                                    "full_graph_info-%d_cat-%d.pkl" % (args.max_depth_infobox,
                                                                       args.max_depth_category))
    full_graph.build_graph(infobox_graph.graph, category_graph.graph)


def build_category_dag(args, category_graph):
    fname_dag = args.results_path + "category_graph_depth-" + \
                str(args.max_depth_category) + "_dag.pkl"
    category_graph.build_dag(fname_dag)


def compute_entities(queries, annotations):
    entities = []
    for _, sample_annotations in annotations.items():
        query = queries[sample_annotations[0].id_]
        entities += query.entities
    entities = list(set(entities))
    return entities


def main():
    args = parse_arguments()

    with open(args.annotations_path + "annotations/queries.pkl", "rb") as fq,\
            open(args.annotations_path + "annotations/annotations.pkl", "rb") as fa:
        queries = pkl.load(fq)
        annotations = pkl.load(fa)
        entities = compute_entities(queries, annotations)

    print("Building Infobox Graph...")
    infobox_graph = build_infobox_graph(args, entities)
    print("Building Category Graph...")
    category_graph = build_category_graph(args, entities)
    print("Building Full RDF Graph...")
    build_full_graph(args, infobox_graph, category_graph)
    print("Building Category DAG...")
    build_category_dag(args, category_graph)

if __name__ == "__main__":
    main()