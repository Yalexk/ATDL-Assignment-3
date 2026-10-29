# ISIC Replication Discussion

## Experiment Scope

This experiment evaluates the ISIC melanoma-segmentation part of *Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation* (Wu et al., 2023). The reported run trained for 40 epochs on three NVIDIA B200 GPUs and produced the following test-set results:

| Metric | Result |
| --- | ---: |
| Validation loss ("Total score") | 0.186749 |
| IoU | 85.56% |
| Dice | 91.70% |

The repository computes "Total score" using a positive-weighted `BCEWithLogitsLoss`, so this value is a loss (lower is better), not a segmentation score. Its evaluation code averages IoU and Dice over output thresholds 0.1, 0.3, 0.5, 0.7, and 0.9. The reported metrics should therefore be described as this implementation's threshold-averaged results, rather than assumed to be directly interchangeable with metrics computed using a single threshold.

## Relation to the Paper

The paper reports a Med-SA one-point result of 92.6% Dice and 84.1% IoU for melanoma segmentation; it also evaluates three-point and two bounding-box prompt settings. Numerically, the present Dice is 0.90 percentage points below the reported one-point Dice, while its IoU is 1.46 points above the reported one-point IoU. These differences are descriptive only and should not be interpreted as a controlled reproduction comparison.

The data protocols differ: the paper describes experiments on ISIC 2019, whereas this repository's ISIC loader reads the ISBI 2016 Part 1 train/test files. The local experiment is thus an ISIC-domain replication using the released code, but not an exact replication of the paper's ISIC dataset.

The prompt protocol is the main experimental limitation. The paper evaluates four conditions: one positive point, three positive points, and bounding boxes with the paper's specified 50% and 75% target-overlap settings. A typical controlled test can construct a box from the ground-truth mask, perturb or adjust its edges to achieve the intended overlap, and supply that box to SAM. This uses the annotation to simulate a user prompt; it is not a box predicted by the segmentation model. The overlap values describe the prompt-generation protocol and should not be implemented by simply multiplying all box coordinates by 0.5 or 0.75. The exact box-generation procedure used for the paper's reported numbers is not recoverable from this checkout alone.

In this repository, the ISIC dataset returns a single randomly sampled positive foreground click. Although `random_box` exists, it jitters mask bounds by a fixed pixel amount rather than accepting the paper's 0.5/0.75 overlap settings, and the ISIC dataset does not return a box. More decisively, the train and validation paths call the SAM prompt encoder with `boxes=None`, so even a box added to a dataset batch would currently be ignored. Likewise, choosing three clicks requires generating and passing three coordinates; reproducing an iterative protocol may also require running the model between clicks and selecting later clicks from its error regions. Neither the click count nor box overlap is configurable from the current test command. Consequently, this run is most comparable to the paper's one-point Med-SA result, not its three-point or bounding-box results. Reproducing those other conditions requires implementing prompt generation and passing the resulting prompts through both evaluation and, where applicable, training.

The paper reports using four NVIDIA A100 GPUs. The hardware in this experiment (three B200 GPUs) is different, but a newer GPU alone does not invalidate a replication: it may affect runtime, and can affect numerical results if precision or nondeterministic operations differ. More importantly, the repository's `-distributed` option defaults to `none`. Confirm that the run explicitly enabled all three devices (for example, with `-distributed 0,1,2`) before stating that training used three GPUs; otherwise the code may have trained only on the selected `-gpu_device`.

## Reproducibility Assessment

The results show strong segmentation performance on the test split used by this run and are broadly in the range of the paper's reported one-point Med-SA result. They do not, by themselves, confirm the paper's ISIC result: the dataset version differs, the repository's metric implementation uses its own threshold-averaging protocol, and the prompt setup only matches the one-point condition approximately. No repeated runs or uncertainty estimates were provided, so run-to-run variation cannot be assessed.

The appropriate conclusion is **partial replication**. The experiment reproduces a 40-epoch, one-click ISIC evaluation using the authors' codebase and obtains a Dice of 91.70%. Exact reproduction remains unverified because the dataset and potentially the evaluation protocol differ from the paper. The result supports the usefulness of the adapted model on this local ISIC split, but stronger claims about reproducing the published ISIC finding require matching the paper's dataset and prompt protocol.

## Details to Record in the Final Report

For full reproducibility, report the exact dataset split and ISIC release, command-line arguments, image size, batch size, SAM encoder/checkpoint, random seed, optimizer and learning rate, and whether `-distributed 0,1,2` was used. Include the precise metric thresholding protocol. If the assignment requires reproducing the paper's prompt ablations, separately implement and evaluate 1-point, 3-point, BBox 0.5, and BBox 0.75 prompts; this checkout's current test path does not implement those alternatives.

## Reference

Wu, J. et al. “Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation.” arXiv:2304.12620, 2023. [https://arxiv.org/abs/2304.12620](https://arxiv.org/abs/2304.12620)

Implementation references: [ISIC dataset loader](dataset/isic.py), [dataset selection](dataset/__init__.py), [validation and prompt handling](function.py), and [evaluation metrics](utils.py).