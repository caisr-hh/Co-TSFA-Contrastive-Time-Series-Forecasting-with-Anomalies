# Co-TSFA: Contrastive Time-Series Forecasting with Anomalies
## Official Codebase for the ICLR 2026 Submission

This repository contains the implementation used in the paper “Contrastive Time-Series Forecasting with Anomalies (Co-TSFA)” 

The method introduces a contrastive regularization framework designed to improve the robustness of time-series forecasting models when evaluated under anomalous conditions.

Co-TSFA is model agnostic and can be applied to any model using an encoder-decoder structure or similar. This repository is based on the Time-Series-Library repository by Tsinghua University, but the files are adapted to support Co-TSFA. The Co-TSFA contrastive loss function is defined under ./utils/contrastive_losses.py and the anomaly injection functions are defined under ./utils/anomaly_injection.py. Since different models use different type of encoders that may consist of multiple steps, Co-TSFA is applied differently for specific models. Therefore, each model has their own exp files under the "exp" folder. 
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```
2. Download the datasets. References to the datasets are in the paper (the ATM dataset is not public).
3. Run one of the command examples or experiment examples in the scripts folder to run the models. Since the ATM dataset won't be made public, the commands for the other datasets are recommended for running the code. 
