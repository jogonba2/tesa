"""
Script to create the modeling task to be solved by models.

Usages:
    tests:
        python create_modeling_task.py -t context-free-same-type --classification --generation --no_save
    regular usages:

        python create_modeling_task.py -t context-free-same-type --classification --generation

        - Generation and classification specifying the task type of the paper
        python create_modeling_task.py -t context-dependent-same-type --classification --generation

        - Generation CF-v0, TF-v2
        python create_modeling_task.py --generation -cf v0 -tf v2

        # Extreme rankings to test robustness (tesa + category graph)
        python create_modeling_task.py -t extreme-ranking --classification -cf v0 -tf v0 --ranking_size 2000 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo extreme_candidates \
                                       --batch_size 1

        # Extreme rankings (with the aggregations of tesa)
        python create_modeling_task.py -t extreme-ranking-tesa --classification -cf v0 -tf v0 --ranking_size 235 \
                                       --batch_size 1


        # Information of the graphs as input #
        - Generation TF-V2 with lowest common ancestors (ABCE) (Category DAG)
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6_dag.pkl \
                                       --category_symbolic_algo lowest_common_ancestors

        - Generation TF-V2 with lowest common ancestors (ABCE) (Category Graph)
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph

        - Ablation study (symbolic info), example with symbolic+context+entities (ACE) (Category DAG)
        python create_modeling_task.py --generation -cf v_ace -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6_dag.pkl \
                                       --category_symbolic_algo lowest_common_ancestors

        - Generation TF-V2, add intersection of infobox values (hardcode *hops* and *k*) (Infobox Graph)
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --infobox_graph_file results/knowledge_graphs/infobox_graph_depth-3.pkl \
                                       --infobox_symbolic_algo value_intersection_infobox

        - Generation TF-V2 with intersection of values at hop-1
          (hardcode *hops* and *k*) (Infobox Graph) and lowest common ancestors (Category Graph)
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --infobox_graph_file results/knowledge_graphs/infobox_graph_depth-3.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph \
                                       --infobox_symbolic_algo value_intersection_infobox

        - Generation TF-V2 with single/group/query information blocks CF={v_blocks1, v_blocks2, ...}
        python create_modeling_task.py --generation -cf v_blocks1 -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --infobox_graph_file results/knowledge_graphs/infobox_graph_depth-3.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph \
                                       --infobox_symbolic_algo value_intersection_infobox

        # Information of the graph as candidates #

        - Generation TF-V2 lowest common ancestors as positive candidates from the category graph.
        The symbolic_format = target, also implies that the distractors are computed as usual (same type)
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph \
                                       --symbolic_format target

        - Generation TF-V2 value intersection as positive candidates from the infobox graph.
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --infobox_graph_file results/knowledge_graphs/infobox_graph_depth-3.pkl  \
                                       --infobox_symbolic_algo value_intersection_infobox \
                                       --symbolic_format target

        - Generation TF-V2 value intersection and lowest common ancestors as positive candidates from
          the infobox and the category graphs.
        python create_modeling_task.py --generation -cf v0 -tf v2 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph \
                                       --infobox_graph_file results/knowledge_graphs/infobox_graph_depth-3.pkl  \
                                       --infobox_symbolic_algo value_intersection_infobox \
                                       --symbolic_format target


        - Negative symbolic information as negative candidates (Discriminative). The 25% of the negative candidates
          are sampled from the graph. (Experiment-1 WeeklyMeeting10)
        python create_modeling_task.py --classification -cf v0 -tf v0 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo negatives_from_neighborhood \
                                       --symbolic_format negative_targets_25

        - Positive symbolic information (ancestors from the category graph) as negative candidates (Discriminative).
          are sampled from the graph. (Experiment-2 WeeklyMeeting10)
        python create_modeling_task.py --classification -cf v0 -tf v0 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph \
                                       --symbolic_format negative_targets_100

        - Soft labels for symbolic information (discriminative)

        python create_modeling_task.py --classification -cf v0 -tf v0 \
                                       --category_graph_file results/knowledge_graphs/category_graph_depth-6.pkl \
                                       --category_symbolic_algo lowest_common_ancestors_graph \
                                       --symbolic_format target \
                                       --soft_labels


        # Including negative candidates in the generative training

        - Negation for negative candidates (it includes also the input in the target)

        python create_modeling_task.py --generation -cf v0 -tf v_neg

        - Negation for negative candidates w/o including the input in the target

        python create_modeling_task.py --generation -cf v0 -tf v_neg2


"""

import modeling.modeling_task as modeling_task
from toolbox.parsers import standard_parser, add_annotations_arguments, add_task_arguments
from toolbox.utils import to_class_name


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_annotations_arguments(ap)
    add_task_arguments(ap)

    return ap.parse_args()


def main():
    """ Creates and saves the modeling tasks. """

    args = parse_arguments()

    task_name = to_class_name(args.task)

    task = getattr(modeling_task, task_name)(ranking_size=args.ranking_size,
                                             batch_size=args.batch_size,
                                             context_format=args.context_format,
                                             targets_format=args.targets_format,
                                             context_max_size=args.context_max_size,
                                             k_cross_validation=int(args.cross_validation) * args.k_cross_validation,
                                             valid_proportion=args.valid_proportion,
                                             test_proportion=args.test_proportion,
                                             random_seed=args.random_seed,
                                             save=not args.no_save,
                                             silent=args.silent,
                                             results_path=args.task_path,
                                             annotation_task_results_path=args.annotations_path,
                                             graph_files={"category": args.category_graph_file,
                                                          "infobox": args.infobox_graph_file},
                                             symbolic_algos={"category": args.category_symbolic_algo,
                                                             "infobox": args.infobox_symbolic_algo},
                                             symbolic_format=args.symbolic_format,
                                             soft_labels=args.soft_labels)

    task.process_data_loaders()

    if task.has_symbolic_info:
        if not "extreme" in task_name.lower():
            task.process_symbolic_info()
        else:
            task.process_extreme_rankings()

    if args.classification:
        task.process_classification_task(args.finetuning_data_path)

    if args.generation:
        task.process_generation_task(args.finetuning_data_path)


if __name__ == '__main__':
    main()
