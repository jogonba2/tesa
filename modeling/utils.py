from numpy import mean, std, where
from random import choice, shuffle
from itertools import chain
from collections import Counter
import torch
import numpy as np


def get_ranks(outputs):
    """
    Returns the ranks according to the outputs (1 for highest grade). Deal with draws by assigning the best rank to the
    first output encountered.

    Args:
        outputs: torch.Tensor, (batch_size, 1 ou 2) tensors outputs.

    Returns:
        torch.Tensor, ranks corresponding to the grades in a line Tensor.
    """

    grades = outputs[:, -1]

    n = len(grades)

    sorter = torch.argsort(grades, descending=True)
    ranks = torch.zeros(n, dtype=torch.long)

    ranks[sorter] = torch.arange(1, n + 1)

    return ranks

def build_alignment(golds, candidates):
    """
    TO COMMENT
    """
    alignment = {}
    for gold in golds:
        alignment[gold] = (None, float("inf"))
        for candidate in candidates:
            dist = levenshtein_distance(gold.split(), candidate.split())
            if dist < alignment[gold][1]:
                alignment[gold] = (candidate, dist)
    return alignment

def filter_alignments(alignments, condition=lambda k: k[1][1]==0.):
    """
    TO COMMENT
    """
    for i, alignment in enumerate(alignments):
        alignments[i] = set([alg[1][0] for alg in alignment.items() if condition(alg)])
    return alignments

def build_aligned_ranking_task(positive_candidates, candidates, ranking_task, gen_scores):
    """
    TO COMMENT
    """
    if not positive_candidates:
        return None

    template = {"entities": ranking_task[0][0]["entities"],
                "entities_type": ranking_task[0][0]["entities_type"],
                "wiki_articles": ranking_task[0][0]["wiki_articles"],
                "nyt_titles": ranking_task[0][0]["nyt_titles"],
                "nyt_contexts": ranking_task[0][0]["nyt_contexts"],
                "choices": candidates,
                "gen_scores": gen_scores}

    # Avoid labeling as 1 multiple times the same match (as the generations are striped before this func,
    # the same generation can appear more than once. Pick the highest ranked. This piece of code only
    # consider exact matches!
    targets = torch.zeros(len(candidates))
    for pos_cand in positive_candidates:
        for i, c in enumerate(candidates):
            if pos_cand == c:
                targets[i] = 1
                break

    return [(template, targets)]

def alignment_funcs(fname):
    if fname=="zero_dist":
        return lambda k: k[1][1] == 0.
    elif fname=="leq_one_dist":
        return lambda k: k[1][1] <= 1.
    elif fname=="leq_one_dist_gt_one_len":
        return lambda k: k[1][1] == 0. or (k[1][1] == 1. and len(k[1][0].split()) > 1)

def get_tesa_entities(data_loaders):
    entities = []
    for data_loader in data_loaders:
        for ranking_task in data_loader:
            for inp, _ in ranking_task:
                entities.append(inp["entities"])
    return set(chain(*entities))

def get_gold_standards(ranking_task):
    """
    Returns a list with the gold standards of a given ranking task.
    """
    targets = list(chain(*[tgt for _, tgt in ranking_task]))
    choices = list(chain(*[inp["choices"] for inp, _ in ranking_task]))
    golds = [pair[0].lower() for pair in zip(choices, targets) if pair[1] == 1]
    return golds

def collapse_graph_information(entities, category_info, infobox_info):
    """
    Join the graph informations for the blocks formats.
    """
    entity_enumeration = ", ".join(entities[:-1]) + " and " + entities[-1]
    connector = "are related to:"
    info_enumeration = ""
    if not category_info and not infobox_info:
        return None
    else:
        if category_info:
            info_enumeration += ", ".join(category_info)
        if infobox_info:
            info_enumeration += " " + ", ".join(infobox_info)

        info_enumeration = " and".join(info_enumeration.rsplit(",", 1))
    collapsed_information = "%s %s %s" % (entity_enumeration.strip(),
                                          connector.strip(),
                                          info_enumeration.strip())

    return collapsed_information


def expand_data_loader(data_loader, idx, candidates, ranking_size):
    for candidate in candidates:
        if len(data_loader[idx]) >= ranking_size:
            break
        template = ({"choices": [candidate],
                     "entities": data_loader[idx][0][0]["entities"],
                     "entities_type": data_loader[idx][0][0]["entities_type"],
                     "wiki_articles": data_loader[idx][0][0]["wiki_articles"],
                     "nyt_titles": data_loader[idx][0][0]["nyt_titles"],
                     "nyt_contexts": data_loader[idx][0][0]["nyt_contexts"]}, torch.tensor([0]))
        data_loader[idx].append(template)


def format_symbolic_target(data_loader, idx, symbolic_info, symbolic_format, soft_targets=False):
    """
    Adds symbolic information as candidates. The target_label is 1 for positive candidates and 0 for negative ones.
    If soft_targets is specified, 2 is used (so, the get_classification_row method of modeling_task can
    distinguish them with a new label used for fairseq for soft-labeling).
    Args:
        ...
        symbolic_format: str, if symbolic_format==target, use symbolic_info as positive candidates. Else,
                              use them as negative candidates.
    """
    if symbolic_format == "target":
        target_label = 1 if not soft_targets else 2
        for sym in symbolic_info:
            n_targets = Counter(list(map(lambda x: x.item(), chain(*[tgt for _, tgt in data_loader[idx]]))))

            if soft_targets:
                n_targets[1] = n_targets[1] + n_targets[2]

            if n_targets[1] >= n_targets[0]:
                break

            valid_batch = choice([j for j, pair in enumerate(data_loader[idx]) if not all(pair[1])])
            neg_idx = choice(where(data_loader[idx][valid_batch][1] == 0)[0])
            data_loader[idx][valid_batch][1][neg_idx] = target_label
            data_loader[idx][valid_batch][0]["choices"][neg_idx] = sym.lower()

    elif "negative_targets" in symbolic_format:
        shuffle(symbolic_info)
        targets = list(chain(*[tgt for _, tgt in data_loader[idx]]))
        choices = list(chain(*[inp["choices"] for inp, _ in data_loader[idx]]))
        golds = [pair[0].lower() for pair in zip(choices, targets) if pair[1] == 1]

        # Remove exact matches between gold standards and negative symbolic info #
        symbolic_info = list(set([s.lower() for s in symbolic_info]).difference(set(golds)))

        n_targets = Counter(list(map(lambda x: x.item(), targets)))
        max_replacements = int(n_targets[0] * int(symbolic_format.split("_")[-1].strip()) / 100.)
        replacements = 0
        visited = {}
        target_label = 0 if not soft_targets else 2

        while replacements < max_replacements and symbolic_info:
            neg = symbolic_info.pop()
            valid_batch = choice([j for j, pair in enumerate(data_loader[idx]) if not all(pair[1])])
            neg_idx = choice(where(data_loader[idx][valid_batch][1] == 0)[0])
            if (valid_batch, neg_idx) not in visited:
                data_loader[idx][valid_batch][1][neg_idx] = target_label
                data_loader[idx][valid_batch][0]["choices"][neg_idx] = neg
                replacements += 1
                visited[(valid_batch, neg_idx)] = True
            else:
                symbolic_info.append(neg)

def format_symbolic_input(data_loader, idx, symbolic_info, context_format, graph_type):
    """
    Adds symbolic information to the input.
    """

    # region SymbolicAblationStudy
    if context_format in ["v_ace", "v_ae"]:
        data_loader[idx][0][0]["wiki_articles"] = [", ".join(symbolic_info)]

    elif context_format in ["v_a_or_b_ce"]:
        if symbolic_info:
            data_loader[idx][0][0]["wiki_articles"] = [", ".join(symbolic_info)]
    # endregion SymbolicAblationStudy

    # region InformationBlocks
    elif context_format in ["v_blocks1", "v_blocks2"]:
        if "category" in data_loader[idx][0][0]:
            prefix_graphs = collapse_graph_information(data_loader[idx][0][0]["entities"],
                                                       data_loader[idx][0][0]["category"],
                                                       symbolic_info)
            if prefix_graphs:
                if context_format == "v_blocks1":
                    graph_article_sep = "."
                elif context_format == "v_blocks2":
                    graph_article_sep = " §"

                data_loader[idx][0][0]["nyt_titles"][0] = "%s%s %s" % (prefix_graphs.strip(),
                                                                       graph_article_sep,
                                                                       data_loader[idx][0][0]["nyt_titles"][0].strip())
        else:
            data_loader[idx][0][0][graph_type] = symbolic_info
    # endregion InformationBlocks

    else:
        data_loader[idx][0][0]["wiki_articles"].insert(0, ", ".join(symbolic_info))

def format_context(ranking_or_inputs, context_format, context_max_size):
    """
    Return the context formated depending on context_format.

    Args:
        ranking_or_inputs: list of (inputs, targets) batches, or just an inputs batch.
        context_format: str, version of the context format to use.
        context_max_size: int, maximum number of tokens in the context.
    """
    if isinstance(ranking_or_inputs, list):  # ranking_or_inputs is a ranking
        inputs, _ = ranking_or_inputs[0]
    else:  # ranking_or_inputs is some inputs
        inputs = ranking_or_inputs

    assert isinstance(inputs, dict)

    context_items = []

    if context_format == "v0":  # (no separation token) wikis articles entities
        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.append(wiki_article)

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend([nyt_title + ':', nyt_context])

        context_items.append(', '.join(inputs['entities']))

    elif context_format in ["v_blocks1", "v_blocks2"]:
        single_group_sep = "µ"
        group_query_sep = "£"

        # Single information block #
        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.append(wiki_article)
        context_items.append(single_group_sep)

        # Group information block #
        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend([nyt_title + ':', nyt_context])
        context_items.append(group_query_sep)

        # Query block #
        context_items.append(', '.join(inputs['entities']))

    elif context_format in ["v1", "v2", "v3", "v4"]:
        # wiki_sep wiki1 wiki_sep wiki2 article_sep article1 article_sep article2 entity_sep entity1 entity_sep ...
        if context_format == "v1":
            wiki_sep = "§"
            article_sep = "£"
            entity_sep = "µ"

        elif context_format == "v2":
            wiki_sep = "Information:"
            article_sep = "Article:"
            entity_sep = "Entity:"

        elif context_format == "v3":
            wiki_sep = "<w>"
            article_sep = "<a>"
            entity_sep = "<e>"

        else:
            wiki_sep = "W"
            article_sep = "A"
            entity_sep = "E"

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.extend([wiki_sep, wiki_article])

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend([article_sep, nyt_title + ".", nyt_context])

        for entity in inputs['entities']:
            context_items.extend([entity_sep, entity])

    elif context_format in ["v_ace", "v_ae", "v_a_or_b_ce"]: # ablation studies based on symbolic information

        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.append(wiki_article)

        if context_format in ["v_ace", "v_a_or_b_ce"]:
            for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
                context_items.extend([nyt_title + ':', nyt_context])

            context_items.append(', '.join(inputs['entities']))

        elif context_format == "v_ae":
            context_items.append(', '.join(inputs['entities']))

    elif context_format == "v_trg": # "Perfect information" analysis
        for wiki_article in inputs['wiki_articles']:
            if wiki_article:
                context_items.append(wiki_article)

        for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
            context_items.extend([nyt_title + ':', nyt_context])

        context_items.append(', '.join(inputs['entities']))

        golds = get_gold_standards(ranking_or_inputs)
        context_items.append("§ " + ", ".join(golds).strip() + " §")

    else:  # ablation studies based on v0
        if context_format == "va":  # no wikis
            for nyt_title, nyt_context in zip(inputs['nyt_titles'], inputs['nyt_contexts']):
                context_items.extend([nyt_title + ':', nyt_context])

            context_items.append(', '.join(inputs['entities']))

        elif context_format == "vb":  # no article
            for wiki_article in inputs['wiki_articles']:
                if wiki_article:
                    context_items.append(wiki_article)

            context_items.append(', '.join(inputs['entities']))

        elif context_format == "vc":  # no wikis and no article
            context_items.append(', '.join(inputs['entities']))

        else:
            raise NotImplementedError("Context format not implemented: %s." % context_format)

    context = " ".join(context_items)

    context_words = context.split()
    l1 = len(context_words)
    if l1 > context_max_size:
        context_words = context_words[-context_max_size:]
        l2 = len(context_words)
        context = " ".join(context_words)

        print("Removing %i tokens from the context." % (l1 - l2))

    return context

def format_choice(choice, targets_format, context=None):
    """
    Returns a formatted choice for the ranking evaluation
    Args:
        choice: str, one choice of a given ranking.
        targets_foramt: str, version of the targets format to use.
        context: str, formatted context of the ranking.
    """

    if targets_format == "v0":
        return choice

    elif targets_format == "v2":
        return choice + " | " + context

    elif targets_format == "v3":
        return context + " | " + choice

    elif targets_format == "v_neg":
        return "are " + choice + " | " + context

    elif targets_format == "v_neg2":
        return "are " + choice

    elif targets_format == "v_neg3":
        return choice + " | " + context

def format_targets(ranking, targets_format,
                   context_format=None,
                   context_max_size=None):
    """
    Returns the generation targets as a list of str, depending on targets_format.

    Args:
        ranking: list of (inputs, targets) batches
        targets_format: str, version of the targets format to use.
        context_format: str, version of the context format to use in the decoder.
        context_max_size: int, max_size of the context to be used in the decoder.
    """

    valid_choices = []
    all_choices = {}
    for inputs, targets in ranking:
        for choice, target in zip(inputs['choices'], targets):
            if target:
                valid_choices.append(choice)
            all_choices[choice] = target

    if targets_format == "v0":  # one target per valid choice
        return valid_choices

    elif targets_format == "v1":  # all valid choices in one target, separated with separation tokens.
        return ["∂ " + " ∂ ".join(valid_choices)]

    # concatenate the input, in the same context format used for the encoder, with the target.
    elif targets_format in ["v2", "v3"]:
        context = format_context(ranking,
                                 context_format=context_format,
                                 context_max_size=context_max_size)

        if targets_format == "v2":
            return [choice + " | " + context for choice in valid_choices]

        elif targets_format == "v3":
            return [context + " | " + choice for choice in valid_choices]

    # negation for considering negative candidates in the generative model
    elif targets_format in ["v_neg", "v_neg2", "v_neg3"]:
        context = format_context(ranking,
                                 context_format=context_format,
                                 context_max_size=context_max_size)

        if targets_format == "v_neg":
            return ["are %s | %s" % (choice, context) if all_choices[choice]
                    else "are not %s | %s" % (choice, context) for choice in all_choices]

        elif targets_format == "v_neg2":
            return ["are %s" % choice if all_choices[choice]
                    else "are not %s" % choice for choice in all_choices]

        else:
            return ["%s | %s" % (choice, context) if all_choices[choice]
                    else "§ %s | %s" % (choice, context) for choice in all_choices]

    else:
        raise Exception("Targets format not implemented: %s." % targets_format)


def list_remove_none(l):
    """
    Removes None from the list l.

    Args:
        l: list, initial list to process.

    Returns:
        list, final list, without None.
    """

    return [item for item in l if item is not None]


def dict_append(d1, d2):
    """
    Append the elements of d2 to the elements of d1.

    Args:
        d1: dict, main dictionary.
        d2: dict, secondary dictionary, appended to the main dictionary.
    """

    for key, item in d2.items():
        d1[key].append(item)


def dict_mean(d):
    """
    Returns a dictionary with the mean of the lists of the dictionary d.

    Args:
        d: dict, input dictionary.

    Returns:
        dict, mean dictionary.
    """

    return {key: mean(item) for key, item in d.items()}


def dict_std(d):
    """
    Returns a dictionary with the standard deviation of the lists of the dictionary d.

    Args:
        d: dict, input dictionary.

    Returns:
        dict, standard deviation dictionary.
    """

    return {key: std(item) for key, item in d.items()}

def get_perplexity(norm_probas):
    return 1. / norm_probas

def levenshtein_distance(a, b, normalized=False):
    M = np.zeros((len(a) + 1, len(b) + 1))

    for i in range(1, len(a) + 1):
        M[i][0] = M[i - 1][0] + 1
    for j in range(1, len(b) + 1):
        M[0][j] = M[0][j - 1] + 1

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            M[i][j] = min(M[i - 1][j] + 1,
                          M[i][j - 1] + 1,
                          M[i - 1][j - 1] + 1 if a[i - 1] != b[j - 1] else M[i - 1][j - 1])

    return M[-1][-1] if not normalized else M[-1][-1] / max(len(a), len(b))