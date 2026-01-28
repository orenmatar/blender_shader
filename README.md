# Learning to Search: Neural-Guided Program Synthesis

**📚 [Read more about the algorithm →](https://orenmatar.github.io/neuro_search_program_synthesis/)**

This project is an exercise in program synthesis focused on generating procedural Blender shader node trees that reproduce a target texture image.

## Quick start
- Requirements: Python 3.8+, Blender (for full image generation), numpy, networkx, PyTorch and other deps in requirements.txt.
- Install project requirements (pip / conda) and ensure Blender is callable if you want to render images.
- Use the notebooks in the repository for examples and interactive runs.

## High level overview
- Goal: start from a simple shader tree and iteratively propose code edits so that rendered textures match a given target image.
- Variations to shader trees are expressed as small edit descriptors. These are applied to BlenderTreeManager instances that hold shader node graphs.
- A code-corrector neural model proposes edits; an MCTS-based or sampling-based loop uses those proposals to explore and find better shaders.

## Core modules
- `Logic/blender_tree_manager.py`
  - Holds a single shader node tree (BlenderTreeManager).
  - Provides creation, serialization (to/from dict and string), random generation, parameter access, and conversion to Blender python code.
  - Useful utilities: copy, compare, calculate free inputs, rename nodes, and generate code to hand to Blender.

- `Logic/tree_networks_manager.py`
  - Manages a directed graph of BlenderTreeManager instances (clusters, sequences, steps).
  - Tracks edges that encode the variation required to go from one tree to another.
  - Supports adding clusters, sequences, connecting nodes, saving/loading the network, and generating images for multiple nodes.

- `Logic/mcts_operator.py`
  - Orchestrates the search towards a target image.
  - Uses a code-corrector NN to propose edits and chooses between greedy, sampling or MCTS-based expansion strategies.
  - Responsible for embedding target images, running evaluation, and optionally optimizing parameters (via Optuna).

- `Logic/NN_models`
  - Neural nets used by the project:
    - code-corrector (BERT-like) that suggests edit labels over the tree string.
    - image embedder model to compute texture similarity.
    - other helper models and training utilities.

- `Logic/variations_creator.py`
  - Helpers to produce structural and non-structural variations (add/remove node/edge, change params/seeds).

## Notebooks and scripts
- `textures_generator.ipynb`
  - Generate random procedural shaders, collect them into a TreesNetworkManager, and render images for the set.

- `BERT_corrector_dataset_generation.ipynb`
  - Create datasets used to train the code-corrector (edits paired with before/after codes).

- `testing_image_to_code.ipynb`
  - Test direct image→code models (different from the code-corrector approach).

- `training_tokenizer.ipynb`
  - Train a tokenizer used by the code-corrector model.

- `applying_search.ipynb`
  - Example workflow demonstrating the MCTSOperator searching for a shader that matches a target texture.

## Training scripts (Logic/training_scripts)
Brief descriptions of the main training entry-points and the models they train:

- `code_corrector_training.py`
  - Trains the BERT-like code-corrector model that proposes edits to shader tree code (token-level heads + distance prediction).
  - Uses the datasets produced by the corrector dataset notebook, image embeddings, and logs/exports checkpoints (Neptune integration).

- `image_to_code_training.py`
  - Trains the image→code decoder (Transformer) that decodes an image embedding into a shader code sequence.
  - Uses an image embedder and tokenizer; suitable for direct image→code experiments and evaluation.

- `image_embedder_training.py`
  - Trains a siamese image embedder (VGG / ResNet / distance-based variants) to produce texture embeddings.
  - Trained with contrastive/cosine losses to allow comparing generated textures to target images.

- `mcts_training.py` and `mcts_training_utils.py`
  - Uses MCTSOperator to run self-play: the code-corrector proposes edits, MCTS explores, renders textures and produces training examples.
  - Trains the code-corrector via reinforcement-style loop (self-play + replay buffer), similar in spirit to AlphaGo's data-generation + training.

Notes
- These scripts assume data layout under your BLENDER_SHADER_DATA_PATH (images, datasets, active_models, models). Check the top of each script for data path variables.
- Notebooks complement the scripts: use notebooks for smaller experiments and visualization; use scripts for larger / reproducible training runs.
- For fast iteration, set deterministic seeds (many helpers support numpy seeding) and run on machines with GPUs.

## Testing
- Unit tests are under project_files/tests. Run them with pytest to validate managers, serialization roundtrips, and network operations.

## Where to look next
- Start by exploring BlenderTreeManager methods: `generate_random_tree`, `to_str`/`from_str`, `to_dict`/`from_dict`.
- Inspect tree_networks_manager for how sequences and clusters are built and how variations are applied.
- Review the notebooks for end-to-end examples and visual inspection.
