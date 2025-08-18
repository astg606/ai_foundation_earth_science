# AI Foundation Models in Earth Science
## Python Frameworks Needed

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
They are meant to automatically capture complex patterns and inherent features from data.
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

The success of FMs is tied to the quantity of the training data
the used. 
They require a large amount of high-quality, unbiased data to operate. 
The can produce biased or incorrect information due to biases in their
training data (Hadid, Chakraborty and Busby, 2024).

- [PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models](https://github.com/VMarsocci/pangaea-bench) - Propose a standardized evaluation
  protocol that incorporates a wide-ranging selection of datasets, tasks,
  resolutions, and sensor types, establishing a robust and widely applicable
  benchmark for Geospatial FMs.


## Python tools needed to use FMs

__PyTorch/TensorFlow__

These frameworks are essential for building, training, and deploying custom AI models, 
including those fine-tuned from foundation models. 
They offer extensive functionalities for neural networks, optimization, and GPU acceleration.

__Xarray/Dask__

Crucial for handling large, multi-dimensional Earth science datasets 
(e.g., climate model outputs, remote sensing data) 
without the requirement of the entire dataset to fit into memory. 
They provide efficient data structures and parallel computing capabilities, 
enabling the processing of data for foundation models.


## Things to consider

- Effective use of FMs in Earth science necessitates robust tools for handling large,
   diverse, and often complex geospatial and temporal datasets.
- Fine-tuning and interpreting FMs for Earth science applications
    requires significant domain knowledge to ensure scientific validity and meaningful results.

## Reference
- Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., Arx, S.,
   Bernstein, M.S., Bohg, J., Bosselut, A., Brunskill, E., et al.
   On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258 (2021)
- Ramachandran R. [AI Foundation Models to Augment Scientific Data and the Research Lifecycle](https://www.earthdata.nasa.gov/news/blog/ai-foundation-models-augment-scientific-data-research-lifecycle), earthdata.nasa.gov, 2023.
- Hadid A., Chakraborty T. and Busby D.
   When geoscience meets generative AI and large language models: Foundations, trends, and future challenges,
  (2024)
   [ https://doi.org/10.1111/exsy.13654]( https://doi.org/10.1111/exsy.13654)
- Zhu X. X., Xiong Z., Wang Y., Stewart A. J. et al. On the Foundations of Earth and Climate Foundation Models.
   CoRR abs/2405.04285 (2024)
   [https://doi.org/10.48550/arXiv.2405.04285](https://doi.org/10.48550/arXiv.2405.04285)
- Sun X., ten Brink T., Carande W. et al., Towards practical artificial intelligence in Earth sciences,
   _Comput Geosci_ __28__, 1305–1329 (2024).
   [https://doi.org/10.1007/s10596-024-10317-7](https://doi.org/10.1007/s10596-024-10317-7)
- Bodnar, C., Bruinsma, W.P., Lucic, A. et al. A foundation model for the Earth system.
   _Nature_ __641__, 1180–1187 (2025).
   [https://doi.org/10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)
- [PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models](https://github.com/VMarsocci/pangaea-bench)
