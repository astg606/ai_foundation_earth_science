# AI Foundation Models in Earth Science

### Introduction

NASA has the world’s largest collection of Earth observation data. 
More datasets are becoming available with varying resolutions,
processing levels, accuracy, and formats, among other properties.
This vast collection is driven by NASA mission to understand our 
planet as a unified system. 
These observations cover all the major disciplines within 
Earth science — land, atmosphere, ocean, cryosphere, and 
human dimensions (Ramachandran, 2023). 
The data, largely unlabeled and vast in volume, can offer novel insights into 
our planet’s systems.

The data is used in forecasting models of the Earth system to make
them more accurate and reliable.
The models are essential for mitigating natural disasters and 
supporting human progress. 
Such tools not only provide crucial early warnings for extreme 
events but are also important for diverse applications ranging 
from agriculture to healthcare to global commerce. 
Modern Earth system predictions rely on complex models 
developed using centuries of accumulated physical knowledge, 
providing global forecasts of diverse variables for weather, 
air quality, ocean currents, sea ice and hurricanes.

Despite their vital role, Earth system forecasting models 
face several limitations. 
They are extremely computationally expensive.
Their complexity, built up over years of development by large teams, 
complicates rapid improvements and necessitates substantial time and 
expertise for effective management. 
Finally, forecasting models incorporate various approximations, 
such as those for sub-grid-scale processes, limiting accuracy. 
These challenges open the door for alternative approaches that may 
offer enhanced performance (Bodnar et al., 2025).

AI can play a critical role to address the above issues and 
to improve the discovery, access, and use of scientific data. 
AI needs to be included into internal model processes 
for more accurate predictions (with less computational requirements)
and for accelerating scientific discovery.

## Foundation Models

Bommasani et al. define a foundation model (FM) as 
“any model that is trained on broad data (generally using self-supervision at scale) 
that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks” in order 
to study this phenomenon.
FMs are trained on extensive, diverse, and frequently multimodal datasets, 
enabling them to generalize effectively across various tasks, regions, and sensors.
In the context of Earth Science, FMs are often multimodal, having been trained 
on satellite imagery, radar data, topography, weather, and even text-based reports.
They learn the underlying patterns and characteristics of the input data,
and are meant to automatically capture complex patterns and inherent features from data.
They have enormous potential in advancing Earth and climate sciences.
They can be used to streamline the processing of Earth's vast data, reduce the 
complexity of forecastings models while improving their prediction accuracy and 
computational efficiency.
FMs can uncover intricate patterns in weather systems and provide more 
reliable predictions across different time scales.

FMs can revolutionize our understanding of the Earth system towards its 
sustainable management: (Zhu et al., 2024)

- FMs have the ability to harness the vast, often unlabeled, reservoirs of observations and
   climate data by means of learning general representations from them.
   FMs can learn task-agnostic general feature representations from massive amounts of unlabelled data.
   This capability not only improves our understanding of Earth’s systems, but also accelerates the pace of
   discovery from the learned general features.
- FMs are trained through self-supervised learning on large, unlabeled datasets,
   which can then be fine-tuned with smaller, task-specific datasets,
   reducing the reliance on extensive labeled data.
   This efficiency not only accelerates the applications of AI for observations,
   but also improves their accuracy, robustness and reliability.
- Earth Science FMs provide a framework through which observation data can directly
   inform and enhance climate models.
   This is essential for improving the resolution and prediction accuracy of
   climate models, and reducing their biases and uncertainty,
   and thus leads to more effective and sustainable climate change mitigation
   and adaptation strategies.
- The complexity and scale of Earth system modeling (ESM) pose challenges to
   computational capacity and efficiency with general circulation models and
   numerical weather prediction models.
   FMs offer scale invariance and efficient feature representations,
   presenting an opportunity for a unified approach to ESM.
   FMs show promise in analyzing diverse ESM data modalities,
   including images and sequences, to facilitate tasks such as physical law discovery. 

![fig_foundationmodel](https://arxiv.org/html/2405.04285v1/x1.png)
Image Source: https://arxiv.org/html/2405.04285v1

Figure 1: _The scheme of an Earth and climate FM. It should be trained on common data modalities, including imagery (radar, optical), non-Euclidean data (point clouds, text), and meteorological data. It should also provide consistency with physical laws. The FM is task-agnostic, exemplified by five possible downstream tasks. Because of the large difference of the characteristics among Earth observation, weather, and climate data, the resulting FM may consists of a pool of expert models, where feedback loops exist among each of them._ 

For the Earth Science research communities, the involvement
of AI is no doubt a huge transition from the traditional
physics-informed numeric models to primarily data-driven AI models. 
Scientists will find the AI prediction less interpretable
than numeric models as AI directly learned all the
patterns from the data instead of pre-fixed physics equations.
However, AI approaches can also strengthen traditional
process-based models by effectively uncovering previously
unknown relations between variables or processes (Sun, 2024).
Scientists need to work in a hybrid environment where 
numerical models and AI models coexist, and their relationship
are interdependent.

AI-enhanced physics models, achieved through the creation
of hybrid physics-AI models involve the incorporation of AI
models into existing physics models. AI models can
replace one or more components of a physics-based model
or predict intermediate quantities that are inadequately 
represented by physics alone (Sun, 2024).


A very possible situation may be that AI models will rely on 
obersvations and numerical model results for training, 
while numerical models can use AI models to skip
some computation-expensive steps.

## Evaluating FMs

The success of FMs is tied to the quantity of the training data they used. 
They require a large amount of high-quality, unbiased data to operate.
FMs generalize well with high-quality training data, 
but they can also magnify biases and inconsistencies present in the data. 
They can produce biased or incorrect information due to biases in their
training data.
All these issues are important to be taken into consideration to 
effectively capitalize and guarantee the beneficial use FMs (Hadid, Chakraborty and Busby, 2024).

### Evaluation aspects

Myren and Parikh, 2025, identify three fundamental evaluation aspects for AI that are
applicable to FMs:

- __Performance uncertainty__:
   - _Training uncertainty_: Two identical AI model architectures with identical
       training approaches exhibit different performances as a result of different
       initializations and random batching during training.
   - _Data uncertainty_: The data sample and how data is split data into training,
       validation, and test partitions have drastic random effects on
       performance benchmarking.
- __Learning efficiency__: The relationship between the gain in performance to the
   additional training data required.
    It is particularly important to FMs because they are fine-tuned using smaller,
    targeted datasets for the task at hand, and the FM that best adapts may be preferred.
- __Overlap in the training and test data__: An overlap between training and test data is
    an example of data leakage that can lead to overestimation of model performance.
    To address this, we need to clearly partition the data into train and test sets
    such that evaluation on the test data reflects model’s performance on truly
    new (or unseen) data.
    Ensuring no overlap may be difficult for FMs, because a major component of
    the "foundation" in FMs is the big data used for pre-training that is often
    scraped from any available source.

In addtion to the above considerations, we need to ensure that FMs are scientifically validated and their results are reproducible to support reliable and trustworthy Earth Science research. 
A possible step towards trustworthyness is to develop explainable AI (models capable of generating decisions that a human could understand and interpret).


### Evaluation tools

Highly accurate FMs can still produce results that are biased or 
difficult to interpret, raising the need for evaluation frameworks that 
consider robustness, fairness, and explainability.
We currently do not have rigorous evaluation frameworks of FMs in Earth Science.
It is therefore difficult to assess FMs suitability for various tasks. 
There is a need for comprehensive and standardized benchmarks for
evaluating FMs.

Here are some evaluation tools:

- [PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models](https://github.com/VMarsocci/pangaea-bench) - Propose a standardized evaluation
  protocol that incorporates a wide-ranging selection of datasets, tasks,
  resolutions, and sensor types, establishing a robust and widely applicable
  benchmark for Geospatial FMs.
- [Awesome Remote Sensing Foundation Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models) - A collection of papers, datasets, benchmarks, code, and pre-trained weights for Remote Sensing Foundation Models (RSFMs).
- [GEO-Bench](https://proceedings.neurips.cc/paper_files/paper/2023/file/a0644215d9cff6646fa334dfa5d29c5a-Paper-Datasets_and_Benchmarks.pdf) - Propose a benchmark comprised of
  six classification and six segmentation tasks, which were carefully curated
  and adapted to be both relevant to the field and well-suited for model evaluation.
  The benchmark includes a robust methodology for evaluating models and reporting
  aggregated results to enable a reliable assessment of progress.
- [PhilEO Bench: Evaluating Geo-Spatial Foundation Models](https://arxiv.org/abs/2401.04464) - Propose a framework
   consisting of a testbed and a novel 400 GB Sentinel-2 dataset containing
   labels for three downstream tasks, building density estimation,
   road segmentation, and land cover classification.
- Duderstadt, Helm and Priebe (2024) propose a methodology for directly comparing
  the embedding space geometry of FMs, which facilitates model comparison without
  the need for an explicit evaluation metric. 
- Geospatial Explainable AI (XAI) - Increase the accuracy and transparency of AI models and to
make their results interpretable. The benchmark contains separate tasks that allows the
user to test a FM’s properties in the embedding space, and demonstrate whether
the model has learned spectral, spatial and temporal features (Alemohammad et al., 2025).


## Python tools needed to use FMs

### Frameworks

- __TensorFlow__: An open-source deep learning framework developed by Google.
   It's renowned for its flexibility and scalability, making it suitable for many AI applications. 
- __PyTorch__: An open-source machine learning library known for its dynamic computational graph.
   The framework is excellent for prototyping and experimentation. 
 
They offer extensive functionalities for neural networks, optimization, and GPU acceleration.

### Reading data

- __Xarray/Dask__: Crucial for handling large, multi-dimensional Earth science datasets 
(e.g., climate model outputs, remote sensing data) without the requirement of the entire dataset to fit into memory. They provide efficient data structures and parallel computing capabilities, 
enabling the processing of data for foundation models.

## Prithvi-WxC Foundation model

- Prithvi (name derived from the Sanskrit word for Earth) is a transformer-based geospatial FM pre-trained on more than 1TB of multispectral satellite imagery from the Harmonized Landsat-Sentinel 2 (HLS) dataset (Jakubik, 2023).
   - Prithvi was shown to perfom well on on fine-tuning tasks in areas ranging from multi-temporal cloud gap imputation, flood mapping, wildfire scar segmentation, and multi-temporal crop segmentation.
- In the Fall of 2024, NASA and IBM released Prithvi-weather-climate (Prithvi-WxC), a weather and climate foundation model (FM) pre-trained on Modern-Era Retrospective analysis for Research and Applications, Version 2 (MERRA-2) data from NASA's Global Modeling and Assimilation Office to replicate atmospheric dynamics while being capable of dealing with missing information (Schmude, 2024, Koehl, 2024).
- Prithvi-WxC features a flexible architecture, a scalable, hierarchical attention mechanism, and task-independent pretraining. It has 320 million (M) parameters, 220 M as encoder and 100 M as decoder. It implements a hierarchical two-dimensional vision transformer architecture that scales to large token counts.
- Prithvi-WxC facilitates reseach works in areas such as etecting and predicting severe weather and natural disasters, creating targeted forecasts from localized observations, enhancing global climate simulations down to regional levels, and improving the representation of physical processes in weather and climate models.
- 

## Things to consider

- Effective use of FMs in Earth science necessitates robust tools for handling large,
   diverse, and often complex geospatial and temporal datasets.
- Fine-tuning and interpreting FMs for Earth science applications
    requires significant domain knowledge to ensure scientific validity and meaningful results.

The future of FMs lies in their continued evolution toward more integrated
data-driven modeling approaches, offering a more comprehensive understanding of Earth’s complex systems. This
progression promises to provide critical insights into pressing global issues, such as climate change, natural hazards, and sustainability, ultimately transforming our approach
to Earth system science and informing decision making in the face of global environmental challenges
(Zhang, 2024).

## Reference
- Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., Arx, S.,
   Bernstein, M.S., Bohg, J., Bosselut, A., Brunskill, E., et al.
   On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258 (2021)
- Ramachandran R. [AI Foundation Models to Augment Scientific Data and the Research Lifecycle](https://www.earthdata.nasa.gov/news/blog/ai-foundation-models-augment-scientific-data-research-lifecycle), earthdata.nasa.gov, 2023.
- Hadid A., Chakraborty T. and Busby D.
   When geoscience meets generative AI and large language models: Foundations, trends, and future challenges,
  Expert Systems __41__(10) (2024)
   [https://doi.org/10.1111/exsy.13654]( https://doi.org/10.1111/exsy.13654)
- Zhu X. X., Xiong Z., Wang Y., Stewart A. J. et al. On the Foundations of Earth and Climate Foundation Models.
   CoRR abs/2405.04285 (2024)
   [https://doi.org/10.48550/arXiv.2405.04285](https://doi.org/10.48550/arXiv.2405.04285).
- Sun X., ten Brink T., Carande W. et al., Towards practical artificial intelligence in Earth sciences,
   _Comput Geosci_ __28__, 1305–1329 (2024).
   [https://doi.org/10.1007/s10596-024-10317-7](https://doi.org/10.1007/s10596-024-10317-7).
- Duderstadt B., Helm H. S. and Priebe C. E.,
   Comparing Foundation Models using Data Kernels,
   arXiv:2305.05126v3 (2024)
   [https://doi.org/10.48550/arXiv.2305.05126](https://doi.org/10.48550/arXiv.2305.05126).
- Zhang H.  et al.,
    When Geoscience Meets Foundation Models: Toward a general geoscience artificial intelligence system,
    _IEEE Geoscience and Remote Sensing Magazine_,
    [https://ieeexplore.ieee.org/document/10770814](10.1109/MGRS.2024.3496478).
- Bodnar, C., Bruinsma, W.P., Lucic, A. et al. A foundation model for the Earth system.
   _Nature_ __641__, 1180–1187 (2025).
   [https://doi.org/10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)
- [PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models](https://github.com/VMarsocci/pangaea-bench)
- Alemohamma, H., Khallaghi S., Godwin D., Balogun R., Roy S., and Ramachandran R.,
   An Explainable AI (XAI) Benchmark for Geospatial Foundation Models,
   EGU General Assembly 2025, Vienna, Austria, 27 Apr–2 May 2025, EGU25-3302, (2025)
  [https://doi.org/10.5194/egusphere-egu25-3302](https://doi.org/10.5194/egusphere-egu25-3302)
- Myren S. and Parikh Nidhu,
   Towards Foundation Models: Evaluation of Geoscience Artificial Intelligence with Uncertainty,
   [https://arxiv.org/html/2501.14809v1](https://arxiv.org/html/2501.14809v1)
- Jakubik J., Roy S., Phillips C. E. et al.,
    Foundation Models for Generalist Geospatial Artificial Intelligence,
    arXiv:2310.18660v2 (2023)
    [https://doi.org/10.48550/arXiv.2310.18660](https://doi.org/10.48550/arXiv.2310.18660).
- Schmude J., Roy S., Trojak  W. et al.,
   Prithvi WxC: Foundation Model forWeather and Climate,
   arXiv:2409.13598 (2024)
   [https://doi.org/10.48550/arXiv.2409.13598](https://doi.org/10.48550/arXiv.2409.13598)
- Koehl D., Prithvi-weather-climate: Advancing Our Understanding of the Atmosphere, [Earthdata Blog](https://www.earthdata.nasa.gov/news/blog/prithvi-weather-climate-advancing-our-understanding-atmosphere),
    (2024).
