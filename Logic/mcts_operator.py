import os
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import optuna
import pandas as pd
import torch._dynamo
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import torch.nn.functional as F

from Logic.NN_models.image_embedders import load_resnet_model
from Logic.NN_models.code_corrector_model import CodeCorrector
from Logic.bpy_connector import generate_image
from Logic.constants import (
    SECONDARY_HEAD_ID_TO_LABEL_MAP,
    MAIN_HEAD_ID_TO_LABEL_MAP,
    HEAD2_OPTIONS_BY_HEAD1,
    MAIN_HEAD_LABEL_TO_ID_MAP,
    SECONDARY_HEAD_LABEL_TO_ID_MAP,
)
from Logic.NN_models.data_loaders import CodeCorrectorDataset
from Logic.tree_networks_manager import (
    TreesNetworkManager,
    CODE_PREDICTED_VALUE,
    FAILED_IMAGE_GENERATION,
    TEXTURE_SIMILARITY_VALUE,
    HAS_IMAGE,
    ACTION_PREDICTED_VALUE,
    HAD_POST_IMAGE_EXPANSION,
    VISITS,
    TOTAL_VALUE,
)
from Logic.blender_tree_manager import BlenderTreeManager
from Logic.node_readers_writers import ParamRequestType, ParamType, VECTOR
from Logic.tokens_to_labels_and_back import token_labels_to_variation_steps
from Logic.variations_creator import VariationDescriptor, VariationType

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


@dataclass
class NodeResult:
    node_value: float
    corrections: list[VariationDescriptor]


class MCTSOperator:
    """
    The tree search operator: given a target image and a starting point - it will generate code variations in the
    direction of the target image.
    The main function is search() which will perform the search and return the final network manager and all nodes
    """
    def __init__(self, corrector_model_path: str, tokenizer_path: str, image_embedder_path: str):
        """
        :param corrector_model_path: path of the corrector model
        :param tokenizer_path: tokenizer used by the corrector model
        :param image_embedder_path: image embedder used to compare images and determine how similar their textures are
        """
        self.loader = CodeCorrectorDataset([], None, random_flips=False)
        self.loaded_target_image = None
        self.work_dir = ""
        self.network_manager = TreesNetworkManager(None)
        self.starting_point_nm = BlenderTreeManager()
        self.target_image_texture_embedding = None
        self.start_node_id = "0"
        self.target_is_set = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.code_corrector = CodeCorrector.load(corrector_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.code_corrector.eval()
        self.code_corrector.to(self.device)
        self.image_embedder = load_resnet_model(image_embedder_path)
        self.image_embedder.eval()
        self.image_embedder.to(self.device)

        self.labels1_to_names_mapper = np.array(
            [MAIN_HEAD_ID_TO_LABEL_MAP[i] for i in sorted(MAIN_HEAD_ID_TO_LABEL_MAP)]
        )
        self.labels2_to_names_mapper = np.array(
            [SECONDARY_HEAD_ID_TO_LABEL_MAP[i] for i in sorted(SECONDARY_HEAD_ID_TO_LABEL_MAP)]
        )
        # array with mask options for the second head, given the first head
        # in head2_mask_by_head1[head1_label] you will find an array of 0s and 1s signifying which head2 labels are legit
        self.head2_mask_by_head1 = np.zeros((len(MAIN_HEAD_ID_TO_LABEL_MAP), len(SECONDARY_HEAD_ID_TO_LABEL_MAP)))
        for head1_value, head2_values in HEAD2_OPTIONS_BY_HEAD1.items():
            head1_label = MAIN_HEAD_LABEL_TO_ID_MAP[head1_value]
            for head2_value in head2_values:
                if head2_value in SECONDARY_HEAD_LABEL_TO_ID_MAP:
                    self.head2_mask_by_head1[head1_label, SECONDARY_HEAD_LABEL_TO_ID_MAP[head2_value]] = 1

    def set_new_target(self, target_img_path: str, work_dir: str, starting_point_btm: None | BlenderTreeManager = None):
        """
        sets a new target image for the operator to search towards,
        optionally with a specific starting point (blender tree manager), otherwise starts from a blank network
        """
        if starting_point_btm is None:
            starting_point_btm = BlenderTreeManager()
            starting_point_btm.initialize_network()
            starting_point_btm.finish_network()
        self.network_manager = TreesNetworkManager(work_dir)
        self.start_node_id = self.network_manager.add_cluster(starting_point_btm)
        self.loader = CodeCorrectorDataset([], work_dir, random_flips=False)
        self.loaded_target_image = self.loader.open_image(img_path=target_img_path).to(self.device)
        self.work_dir = work_dir
        self.target_image_texture_embedding = self._embed_images(self.image_embedder, [target_img_path])
        self.target_is_set = True

    def search(
        self,
        target_img_path: str,
        workdir: str,
        search_method="sample",
        temperature=1,
        c_puct=2,
        sample_labels=True,
        optimize=False,
        max_expansions=20,
        max_nodes_to_expand_per_iter=30,
        n_nodes_to_optimize_at_end=3,
    ) -> tuple[TreesNetworkManager, list]:
        """
        The main search function - performs the search towards the target image, starting from the given starting point
        :param target_img_path: path to the target image
        :param workdir: working directory to use for temporary images generated as part of the search
        :param search_method: method to use - "greedy", "sample" or "mcts"
        :param temperature: temperature to use for sampling (only for "sample" method)
        :param c_puct: exploration constant to use for MCTS (only for "mcts" method)
        :param sample_labels: whether to sample labels from the code corrector model or take the max
        :param optimize: whether to optimize the best nodes founds at the end of the search, using optuna for their params
        :param max_expansions: maximum number of expansion iterations to perform
        :param max_nodes_to_expand_per_iter: maximum number of nodes to add to the network at every iteration
        :param n_nodes_to_optimize_at_end: number of best nodes to optimize at the end of the search, if optimize is True
        """
        self.set_new_target(target_img_path, workdir, starting_point_btm=None)
        all_node_expansions = []
        nodes_to_expand = [self.start_node_id]
        for i in range(max_expansions):
            # STEP 0: generate an image for all the nodes to expand (if they don't already have an image)
            nodes_to_generate_image = [
                node_id for node_id in nodes_to_expand if not self.network_manager.node_has_label(node_id, HAS_IMAGE)
            ]
            _, failures = self.generate_images_for_nodes(nodes_to_generate_image)
            self.network_manager.delete_nodes(failures)
            nodes_to_expand = [node for node in nodes_to_expand if node not in failures]
            all_node_expansions.append(nodes_to_expand)  # track the progress

            images_names = []
            codes = []
            # STEP 1: prepare codes of nodes to expand, and generate results
            for node in nodes_to_expand:
                with_image = self.network_manager.node_has_label(node, HAS_IMAGE)
                images_names.append(node if with_image else None)
                assert all(images_names)  # for now search will only work with images
                nm = self.network_manager.blender_tree_managers[node]
                code = nm.to_str(add_image_tokens=True, with_cur_image=with_image, with_seeds=False)
                codes.append(code)

            # STEP 2: apply the code-corrector NN to generate possible edits/actions/corrections, and their probabilities
            nodes_results = self._codes_to_variations_and_scores(codes, images_names, sample_labels=sample_labels)

            # STEP 3: add all variations to the network and the new values
            for node_id, node_result in zip(nodes_to_expand, nodes_results):
                self.network_manager.set_node_value(node_id, CODE_PREDICTED_VALUE, node_result.node_value)
                if search_method == "mcts":
                    self.backprop_values(node_id, node_result.node_value)
                for variation_desc in node_result.corrections:
                    new_node_id = self.network_manager.add_step(node_id, variation_desc)
                    self.network_manager.set_node_value(new_node_id, ACTION_PREDICTED_VALUE, variation_desc.prob)
                # mark nodes that had a generation after their image was generated (no need to come back to them...)
                # for now this is pointless - they all should have images
                if self.network_manager.node_has_label(node_id, HAS_IMAGE):
                    self.network_manager.add_node_label(node_id, HAD_POST_IMAGE_EXPANSION)

            # STEP 4: get the next nodes to evaluate and expand
            nodes_to_consider = self.get_nodes_to_consider()
            if len(nodes_to_consider) == 0:
                break
            if search_method == "greedy":
                # from the nodes to consider - select who has the best predicted value
                node_expected_values = self.get_nodes_expected_value(nodes_to_consider)
                sorted_prediction = sorted(node_expected_values, reverse=True)
                nodes_to_expand = list(zip(*sorted_prediction))[1][:max_nodes_to_expand_per_iter]
            if search_method == "sample":
                # from the nodes to consider - select who has the best predicted value
                node_expected_values = self.get_nodes_expected_value(nodes_to_consider)
                weights, nodes = list(zip(*node_expected_values))
                weights = np.array(weights)
                scaled_weights = weights ** (1 / temperature)
                probabilities = scaled_weights / scaled_weights.sum()
                n_nodes_to_choose = min(max_nodes_to_expand_per_iter, len(nodes))
                nodes_to_expand = np.random.choice(nodes, size=n_nodes_to_choose, replace=False, p=probabilities)
            if search_method == "mcts":
                nodes_to_expand = self.get_mcts_recommendations(c_puct=c_puct, top_k=max_nodes_to_expand_per_iter)

        # Finally: pick the best nodes and optimize their params
        sorted_nodes = self.network_manager.get_sorted_nodes(sort_by=CODE_PREDICTED_VALUE)
        best_nodes_at_end = list(zip(*sorted_nodes[:n_nodes_to_optimize_at_end]))[1]
        if optimize:
            new_node_ids = []
            for node_id in best_nodes_at_end:
                blender_tree_manager = self.network_manager.blender_tree_managers[node_id]
                # sometimes optuna fails - for nodes that have no parameters to optimize
                # TODO: check and deal with it directly instead of try except
                try:
                    value, variation_desc = self.optimize_blender_tree(blender_tree_manager, n_trials=12)
                    new_node_id = self.network_manager.add_step(node_id, variation_desc)
                    self.network_manager.set_node_value(new_node_id, TEXTURE_SIMILARITY_VALUE, value)
                    new_node_ids.append(new_node_id)
                except:
                    pass

            self.network_manager.generate_images_for_nodes(
                new_node_ids, self.work_dir, override_images=True, print_progress=False, resolution=224
            )
            all_node_expansions.append(new_node_ids)
        # make sure we have images for those best nodes
        best_nodes_with_no_image = [
            node_id for node_id in best_nodes_at_end if not self.network_manager.node_has_label(node_id, HAS_IMAGE)
        ]
        self.generate_images_for_nodes(best_nodes_with_no_image)
        return self.network_manager, all_node_expansions

    def generate_images_for_nodes(self, nodes_to_generate_images: list[str]):
        """
        Generates images for the given nodes (blender tree managers), and sets their texture similarity values to the
        target image.
        """
        failed, empty_count = self.network_manager.generate_images_for_nodes(
            nodes_to_generate_images, self.work_dir, override_images=True, print_progress=False, resolution=224
        )
        failures = []
        for node_id, _ in failed:
            print(f"failed generation: {failed}")
            self.network_manager.add_node_label(node_id, FAILED_IMAGE_GENERATION)
            nodes_to_generate_images.remove(node_id)  # we want only nodes with images
            failures.append(node_id)

        # After generation: compare images to target and set their texture similarity
        if len(nodes_to_generate_images) > 0:
            images_paths = [
                self.network_manager.make_image_path(node, self.work_dir) for node in nodes_to_generate_images
            ]
            scores = self.compare_images_to_target(images_paths).flatten()
            for score, node_id in zip(scores, nodes_to_generate_images):
                self.network_manager.set_node_value(node_id, TEXTURE_SIMILARITY_VALUE, score)
        return nodes_to_generate_images, failures

    @staticmethod
    def _sample_labels_from_logits(logits, threshold=0.02):
        """
        Samples labels from logits with a threshold - any label with probability under the threshold is ignored
        Labels are the actions to be taken, on specific tokens
        """
        probs = F.softmax(logits, dim=-1)
        mask = (probs >= threshold).float()
        masked_probs = probs * mask
        # Re-normalize to ensure they sum to 1 along the last dimension
        masked_probs_sum = masked_probs.sum(dim=-1, keepdim=True) + 1e-8  # avoid divide-by-zero
        renormalized_probs = masked_probs / masked_probs_sum
        flat_probs = renormalized_probs.view(-1, renormalized_probs.size(-1))
        sampled_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled_labels = sampled_flat.view(logits.size(0), logits.size(1))

        sampled_labels_unsqueezed = sampled_labels.unsqueeze(-1)  # shape (B, T, 1)
        # Gather the probabilities for the sampled labels
        sampled_probs = torch.gather(renormalized_probs, dim=-1, index=sampled_labels_unsqueezed).squeeze(-1)
        return sampled_labels, sampled_probs

    @staticmethod
    def _convert_to_labels_values(token_cls, selection_mask=None, sample_labels=True):
        token_cls = token_cls.cpu()
        if selection_mask is not None:  # apply selection of only certain labels
            token_cls = token_cls * selection_mask
        if sample_labels:
            labels, values = MCTSOperator._sample_labels_from_logits(token_cls)
        else:
            softmax_output = F.softmax(token_cls, dim=-1).cpu().numpy()
            labels = np.argmax(softmax_output, axis=-1)
            # Get the highest value (max) along the last dimension - the value of the selected label
            values = np.max(softmax_output, axis=-1)
        return labels, values

    def _codes_to_variations_and_scores(
        self, codes: list[str], images: list[str], sample_labels=True
    ) -> list[NodeResult]:
        """
        Given a list of codes and their current images, uses the code corrector to get suggested variations and scores
        """
        with torch.no_grad():
            current_images = [
                self.loader.open_image(name).to(self.device) if name else self.loaded_target_image for name in images
            ]
            current_images = torch.stack(current_images, dim=0)
            add_cur_images = torch.tensor([image is not None for image in images]).to(self.device)
            target_image = self.loaded_target_image.unsqueeze(0).repeat(len(codes), 1, 1, 1).to(self.device)
            tokenized = self.tokenizer(codes, return_tensors="pt", padding=True)
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            match_value, main_token_cls, secondary_token_cls = self.code_corrector(
                input_ids, attention_mask, target_image, current_images, add_cur_images
            )
            main_labels, main_values = self._convert_to_labels_values(main_token_cls, sample_labels=sample_labels)
            match_value = match_value.cpu().flatten()

        label2_selection_mask = self.head2_mask_by_head1[main_labels]
        secondary_labels, secondary_values = self._convert_to_labels_values(
            secondary_token_cls, selection_mask=label2_selection_mask, sample_labels=sample_labels
        )
        # remove all labels under 0s in attention - in the padded area
        tokens_mask = attention_mask.cpu().numpy()
        main_labels = main_labels * tokens_mask
        secondary_labels = secondary_labels * tokens_mask
        mean_value = np.mean([main_values, secondary_values], axis=0)
        named_labels1 = self.labels1_to_names_mapper[main_labels]
        named_labels2 = self.labels2_to_names_mapper[secondary_labels]

        res = []
        for i in range(len(codes)):
            item_labels1 = named_labels1[i]
            item_labels2 = named_labels2[i]
            item_input_ids = input_ids[i]
            item_tokens = self.tokenizer.convert_ids_to_tokens(item_input_ids)
            item_code = codes[i]
            item_probs = mean_value[i]
            item_full_labels = [a + "__" + b for a, b in zip(item_labels1, item_labels2)]
            variations = token_labels_to_variation_steps(
                item_code,
                item_tokens,
                item_full_labels,
                protected_conversion=True,
                confidence_values=item_probs,
                numeric_params=False,
                ensure_correct_syntax_labels=True,
            )
            res.append(NodeResult(match_value[i].item(), variations))
        return res

    def _embed_images(self, model, images_paths: list[str]):
        images = [self.loader.open_image(img_path=img_path) for img_path in images_paths]
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        # Flatten embeddings to make them 1D (if needed for each image in the batch)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings

    @staticmethod
    def _generate_image_from_params(btm: BlenderTreeManager, kwargs_array, output_dir="/tmp", unique_name=False):
        variation = flat_params_to_variation(kwargs_array)
        new_nm = TreesNetworkManager.apply_variation(btm, variation)

        os.makedirs(output_dir, exist_ok=True)

        base_filename = "img"
        extension = ".png"
        new_file_name = os.path.join(output_dir, f"{base_filename}{extension}")

        if unique_name:
            index = 1
            while os.path.exists(new_file_name):
                new_file_name = os.path.join(output_dir, f"{base_filename}{index}{extension}")
                index += 1

        generate_image(new_nm, new_file_name, resolution=224)
        return new_file_name

    def compare_images_to_target(self, images_paths):
        new_image_emb = self._embed_images(self.image_embedder, images_paths)
        similarity = cosine_similarity(new_image_emb, self.target_image_texture_embedding)
        return similarity

    def _make_optuna_objective_func(self, nm, params_array, output_dir="/tmp", unique_name=False):
        assert self.target_is_set

        def objective(trial):
            try:
                all_kwargs = {}
                for name, (range_vals, param_type) in params_array:
                    if param_type == ParamType.CATEGORICAL:
                        all_kwargs[name] = trial.suggest_categorical(name, range_vals)
                    elif param_type == ParamType.FLOAT:
                        all_kwargs[name] = trial.suggest_float(name, range_vals[0], range_vals[1])
                file_name = self._generate_image_from_params(
                    nm, tuple(all_kwargs.items()), output_dir=output_dir, unique_name=unique_name
                )
                similarity = self.compare_images_to_target([file_name])[0][0]
                return similarity
            except:
                return float("nan")

        return objective

    def optimize_blender_tree(self, btm: BlenderTreeManager, n_trials=12):
        """
        takes an BlenderTreeManager and without changing structure - optimizes it to the target by only changing
        numerical and categorical parameters.
        TODO: if we already know the value of the start position it is possible to give it to optuna as a start
        """
        params_array = get_btm_params_as_flat_array(btm, return_ranges=True)
        start_state = get_btm_params_as_flat_array(btm, return_ranges=False)
        study = optuna.create_study(direction="maximize")
        # force it to try the start state as the first trial
        # there may be a small bug, and sometimes it ignore some of the params (vectors?) but it still helps
        study.enqueue_trial(dict(start_state))
        objective = self._make_optuna_objective_func(btm, params_array)
        study.optimize(objective, n_trials=n_trials)
        value = study.best_value
        variation = flat_params_to_variation(study.best_params.items())
        return value, variation

    def get_nodes_to_consider(self):
        # only nodes that have not gone through the code corrector + value giver yet, or did once but now have a new image
        nodes_to_consider = []
        for node_id in self.network_manager.network.nodes:
            node = self.network_manager.network.nodes[node_id]
            if node.get(CODE_PREDICTED_VALUE) is None or (
                self.network_manager.node_has_label(node_id, HAS_IMAGE)
                and not self.network_manager.node_has_label(node_id, HAD_POST_IMAGE_EXPANSION)
            ):
                nodes_to_consider.append(node_id)
        return nodes_to_consider

    def backprop_values(self, leaf_node, value):
        nodes_on_path = nx.shortest_path(self.network_manager.network, source=self.start_node_id, target=leaf_node)
        for node_id in nodes_on_path:
            new_node_visits = self.network_manager.get_node_value(node_id, VISITS, default=0) + 1
            new_node_total = self.network_manager.get_node_value(node_id, TOTAL_VALUE, default=0) + value
            self.network_manager.set_node_value(node_id, VISITS, new_node_visits)
            self.network_manager.set_node_value(node_id, TOTAL_VALUE, new_node_total)

    @staticmethod
    def get_puct(total_value, visit_count, parent_visit_count, prior, c_puct):
        if visit_count == 0:
            value = 100  # like inf, but use exploration to break ties
        else:
            value = total_value / visit_count
        exploration = prior * (parent_visit_count**0.5) / (1 + visit_count)
        return value + c_puct * exploration

    def get_mcts_recommendations(self, c_puct=2, top_k=5):
        current_nodes_to_examine = [((0,), self.start_node_id)]
        leafs = []
        while len(current_nodes_to_examine) > 0:
            nodes_to_calc_puct = []
            for parent_puct, node in current_nodes_to_examine:
                children = list(self.network_manager.network.successors(node))
                if len(children) == 0:  # the parent is a leaf
                    leafs.append((parent_puct, node))
                else:
                    parent_count = self.network_manager.get_node_value(node, VISITS)
                    for child in children:
                        child_count = self.network_manager.get_node_value(child, VISITS, default=0)
                        child_value = self.network_manager.get_node_value(child, TOTAL_VALUE, default=0)
                        child_prior = self.network_manager.get_node_value(child, ACTION_PREDICTED_VALUE)
                        child_puct = self.get_puct(child_value, child_count, parent_count, child_prior, c_puct=c_puct)
                        # the puct tuple keeps track of the puct of all ancestors - so when we sort by it, good puct for ancestors gets priority
                        # this is to mimic the behaviour of a non parallelize puct - which only explores one node at at time
                        nodes_to_calc_puct.append((parent_puct + (child_puct,), child))
            current_nodes_to_examine = sorted(nodes_to_calc_puct, reverse=True)[:top_k]
        return [node_id for _, node_id in sorted(leafs, reverse=True)[:top_k]]

    def get_nodes_expected_value(self, nodes_to_consider):
        node_expected_values = []
        for node_id in nodes_to_consider:
            # every node should have excatly one predecessor, except from the cluster start
            predecessors = list(self.network_manager.network.predecessors(node_id))
            if len(predecessors) == 0:
                continue
            predecessor = predecessors[0]
            predecessor_val = self.network_manager.network.nodes[predecessor].get(CODE_PREDICTED_VALUE)
            node_action_val = self.network_manager.network.nodes[node_id].get(ACTION_PREDICTED_VALUE)
            predecessor_val = max(0.1, predecessor_val)
            expected_value = predecessor_val**2 * node_action_val
            # give more weight to the predecessor expectation by taking the power2 of it
            node_expected_values.append((expected_value, node_id))
        return node_expected_values


def get_btm_params_as_flat_array(btm: BlenderTreeManager, return_ranges=True):
    """
    Converts a BlenderTreeManager to a flat array of its params with all their values
    if return_ranges - returns the range of values for each param, else: returns the actual values
    """
    flat_param_options = []
    nm_params = btm.get_all_nodes_values(ParamRequestType.NON_SEED, return_ranges=return_ranges)
    if return_ranges:
        for node_name, values in nm_params.items():
            for arg_name, (vals, param_type) in values.items():
                if param_type == ParamType.VECTOR:
                    for i in range(3):
                        flat_param_options.append((f"{node_name}___{VECTOR}_{i}&{arg_name}", (vals, ParamType.FLOAT)))
                else:
                    flat_param_options.append((f"{node_name}___{arg_name}", (vals, param_type)))
    else:
        for node_name, values in nm_params.items():
            for arg_name, vals in values.items():
                if isinstance(vals, (list, tuple)):
                    for i in range(3):
                        flat_param_options.append((f"{node_name}___{VECTOR}_{i}&{arg_name}", vals[i]))
                else:
                    flat_param_options.append((f"{node_name}___{arg_name}", vals))
    return tuple(sorted(flat_param_options))


def flat_params_to_variation(flat_params: list) -> VariationDescriptor:
    """
    Converts a flat array of params to the nested-dict (ny node name) needed to update those params in an nm
    """
    params = defaultdict(dict)
    for name, val in flat_params:
        node_name, arg_name = name.split("___")
        if arg_name.startswith(VECTOR):
            arg_name = arg_name.replace(f"{VECTOR}_", "")
            i, name = arg_name.split("&")
            if name not in params[node_name]:
                params[node_name][name] = [0, 0, 0]  # initialize the vector values
            params[node_name][name][int(i)] = val
        else:
            params[node_name][arg_name] = val
    variation = VariationDescriptor(variation_type=VariationType.CAT_AND_NUMERIC, step=dict(params))
    return variation


def normalized_entropy(x):
    x = np.array(x)
    total = np.sum(x)
    if total == 0:
        return 0  # no entropy if nothing to distribute
    p = x / total
    p = p[p > 0]  # avoid log(0)
    entropy = -np.sum(p * np.log2(p))
    return entropy / np.log2(len(x))  # normalize to [0,1]


def search_metrics(network_manager: TreesNetworkManager, all_node_expansions, start_node_id, target_nm=None):
    """
    Given a graph manager after a search, computes various metrics about the search process
    """
    G = network_manager.network
    res = {}
    last_expansion = all_node_expansions[-1]
    final_scores = network_manager.get_sorted_nodes(TEXTURE_SIMILARITY_VALUE)
    best_score = final_scores[0][0]
    around_best_score = best_score - 0.01
    best_nodes = [node_id for score, node_id in final_scores if score > around_best_score]
    step_numbers = [next(i for i, nodes in enumerate(all_node_expansions) if n in nodes) for n in best_nodes]
    distances = [nx.shortest_path_length(G, start_node_id, node) for node in best_nodes]
    best_distance = min(distances)
    res["best_score"] = best_score
    res["best_distance"] = best_distance
    res["best_step"] = min(step_numbers)
    res["best_in_last_expansion"] = max([score for score, node_id in final_scores if node_id in last_expansion])
    res["best_before_last_expansion"] = max([score for score, node_id in final_scores if node_id not in last_expansion])
    res["original_complexity"] = None
    if target_nm is not None:
        res["original_complexity"] = len(target_nm.network.nodes) - 2

    # correlation between code scores by model and the texture score (ground truth)
    all_nodes_scores = [
        (node_data.get(CODE_PREDICTED_VALUE), node_data.get(TEXTURE_SIMILARITY_VALUE))
        for _, node_data in network_manager.network.nodes(data=True)
        if node_data.get(CODE_PREDICTED_VALUE)
    ]
    res["code_texture_corr"] = pd.DataFrame(all_nodes_scores).corr()[0][1]

    # the repetativeness of the last few steps - high means we are kind stuck trying the same moves
    # looking at the last few steps - are there variations that occur all the time (so the search is not really advancing)
    final_edges_steps = []
    for expansion in all_node_expansions[-5:]:
        edges_ending_at_targets = [
            data["variation_type"] + "__".join([str(x) for x in data["step"].items()])
            for _, edge_to, data in G.edges(data=True)
            if edge_to in expansion
        ]
        final_edges_steps.extend(edges_ending_at_targets)

    res["n_repetitions_at_end"] = pd.Series(final_edges_steps).value_counts().iloc[:2].tolist()

    #
    nodes_by_distance = {target: nx.shortest_path_length(G, start_node_id, target) for target in G.nodes}
    leaf_nodes = [n for n in G.nodes if G.out_degree(n) == 0]
    distribution_by_distances = {node_id: 0 for node_id in G.nodes}
    nodes_to_take = [node for node in leaf_nodes if nodes_by_distance[node] > 0]
    for leaf in nodes_to_take:
        # up to the [:-1] - don't include the leaf node, just count how many times passed through
        for node_in_path in list(nx.all_simple_paths(G, source=start_node_id, target=leaf))[0][:-1]:
            distribution_by_distances[node_in_path] += 1

    df = pd.DataFrame(distribution_by_distances.items(), columns=["node_id", "passed_through"])
    df["distance_from_base"] = df["node_id"].apply(lambda x: nx.shortest_path_length(G, start_node_id, x))
    leaf_source_distribution = df.groupby("distance_from_base").passed_through.apply(list)
    res["mean_entropy"] = leaf_source_distribution.iloc[2:-1].apply(normalized_entropy).mean()
    return res
