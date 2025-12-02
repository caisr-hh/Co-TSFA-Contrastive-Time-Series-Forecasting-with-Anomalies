# Co-TSFA: Contrastive Time-Series Forecasting with Anomalies
## Official Codebase for the ICLR 2026 Submission

This repository contains the implementation used in the paper “Contrastive Time-Series Forecasting with Anomalies (Co-TSFA)” 

The method introduces a contrastive regularization framework designed to improve the robustness of time-series forecasting models when evaluated under anomalous conditions.

## Overview

Real-world time series often contain anomalous events that disrupt normal temporal patterns. Some anomalies are short-lived and should be ignored, while others persist and should influence the forecast. Standard models fail to distinguish between these cases.

Co-TSFA addresses this challenge by:

Generating input-only anomalies (irrelevant noise that should not affect the forecast)

Generating input–output anomalies (persistent shifts requiring forecast adaptation)

Introducing a latent–output alignment loss that enforces consistency between changes in the forecast and changes in latent representations

Training forecasting models to be invariant to irrelevant input disturbances while remaining sensitive to meaningful distributional shifts

Co-TSFA is model-agnostic and can be applied to forecasting architectures such as TimesNet, TimeXer, Autoformer, Informer, and iTransformer.
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```
2. Download the datasets. References to the datasets are in the paper.
3. Run one of the command examples or experiment examples in the scripts folder to run the models. Since the ATM dataset won't be made public, the commands for the other datasets are recommended for running the code. 
