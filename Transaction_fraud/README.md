# Credit Card Fraud Detection Models

Kaggle notebook: https://www.kaggle.com/ekaterinadranitsyna/spotting-credit-card-fraud-auc-0-98

Kaggle dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

This repository contains code examples to train and cross-validate **binary classification models for credit card fraud detection**. Several **sklearn classifiers** (Gaussian Naive Bayes, AdaBoost and SVC) and the popular **gradient boosting models** (XGBoost, LGBM and CatBoost) are tested with default parameters. The most promising models are implroved with Grid-Search and hyper-parameter optimization to get higher **AUC score**.

The dataset contains **credit card transactions made by European cardholders in two days** in September 2013: a total of **492 frauds out of 284,807 transactions**. Frauds account for **0.172%** of all transactions, which makes the data set highly imbalanced.

**Features:**
- V1, V2 … V28 - numerical features, described in the data set annotation as the output of PCA
- Time - seconds elapsed between each transaction and the first transaction in the data set
- Amount - the transaction amount

**Target value:**
- Class - takes value 1 in case of fraud and 0 otherwise

![image.png](https://www.kaggleusercontent.com/kf/80919437/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..DotO8bbfANyqzsfq6vhhpA._AMeyFM5qZCA1nGbqUcQDUcwRp1SJJ-iQHGh5m_ZJNhwmxnI4ivusLZtx9Ozyu5oebEOxCcmlzOCrZzeFmKusxWfWWgxNoGrByMJsvoK4Z730qprl5cuBLGuitPETaRCMc-TTkDpETcqmi55trXbojfiGmOCZ2XL72LLQ24HzGPhMnmcB8k79mgODsqsspF1_0NzktvzX-rEsJK6OktdTlp_ISEljkptGWpq5mrBbStw25maHmvdmimZ1iQbJnMrNwxm_m7eHi_NZ2FVSzldR7ePkFM5gql0uAugrkYOpbPix2bSxDrIaD0QCOBHuc2eUfCf40J09ej9eWtPhksxP0LCJxmpDrYQkodHQLFyTwpitX3RhTtOpqum0aOhu6G8rMiHUPtKRhnEHMfljA7HcxNwmfw2SggtGC5B3yX-2tVXH9_bN7qAkfHpApfzOfTBVQd-FqSGA1-XpovyxO3gLDZMtu3YQsOKJGU8FstZx8VknX4e57z_hsbmFAr-FW-KxIwPXWg2GWM3bpU5aABIukmUFDvrIJ_hdVFbdetrgduCslj_u3KKOFRfoS16jBMOikhUNBe0VPCRWpFf_Cv33UaeVAtReAYKOfC7OsY5RtMCWsVaVVlagAYApHJxHVcSV_RzmdZSeSpUYvDCZSGQB7gsnqaKBei6dgE53hADDnJdIuoteEOnf_Cp4IHN_90z.5iqCuk5MimdQ0iCZQN3SnQ/__results___files/__results___12_0.png)

All original features were used for modelling as well as a **new feature (hour of transaction)**, which was extracted from the "Time" column. 

![image.png](https://www.kaggleusercontent.com/kf/80919437/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..DotO8bbfANyqzsfq6vhhpA._AMeyFM5qZCA1nGbqUcQDUcwRp1SJJ-iQHGh5m_ZJNhwmxnI4ivusLZtx9Ozyu5oebEOxCcmlzOCrZzeFmKusxWfWWgxNoGrByMJsvoK4Z730qprl5cuBLGuitPETaRCMc-TTkDpETcqmi55trXbojfiGmOCZ2XL72LLQ24HzGPhMnmcB8k79mgODsqsspF1_0NzktvzX-rEsJK6OktdTlp_ISEljkptGWpq5mrBbStw25maHmvdmimZ1iQbJnMrNwxm_m7eHi_NZ2FVSzldR7ePkFM5gql0uAugrkYOpbPix2bSxDrIaD0QCOBHuc2eUfCf40J09ej9eWtPhksxP0LCJxmpDrYQkodHQLFyTwpitX3RhTtOpqum0aOhu6G8rMiHUPtKRhnEHMfljA7HcxNwmfw2SggtGC5B3yX-2tVXH9_bN7qAkfHpApfzOfTBVQd-FqSGA1-XpovyxO3gLDZMtu3YQsOKJGU8FstZx8VknX4e57z_hsbmFAr-FW-KxIwPXWg2GWM3bpU5aABIukmUFDvrIJ_hdVFbdetrgduCslj_u3KKOFRfoS16jBMOikhUNBe0VPCRWpFf_Cv33UaeVAtReAYKOfC7OsY5RtMCWsVaVVlagAYApHJxHVcSV_RzmdZSeSpUYvDCZSGQB7gsnqaKBei6dgE53hADDnJdIuoteEOnf_Cp4IHN_90z.5iqCuk5MimdQ0iCZQN3SnQ/__results___files/__results___20_0.png)

### Cross-Validation Results

3-fold testing of the classifiers with default parameters (AUC score):
- Gaussian Naive Bayes: 0.960
- AdaBoost: 0.972
- SVC: 0.955
- XGBoost: 0.983
- LGBM: 0.913
- CatBoost: 0.983

### Grid-Search / Parameter Optimization Results

- Gaussian Naive Bayes: 0.967
- AdaBoost: 0.975
- XGBoost: 0.985
- CatBoost: 0.988

Models assigned different importance to input features, which suggests that an ensemble including several classifiers could be a more reliable solution to fraud detection.

![image.png](https://www.kaggleusercontent.com/kf/80919437/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..DotO8bbfANyqzsfq6vhhpA._AMeyFM5qZCA1nGbqUcQDUcwRp1SJJ-iQHGh5m_ZJNhwmxnI4ivusLZtx9Ozyu5oebEOxCcmlzOCrZzeFmKusxWfWWgxNoGrByMJsvoK4Z730qprl5cuBLGuitPETaRCMc-TTkDpETcqmi55trXbojfiGmOCZ2XL72LLQ24HzGPhMnmcB8k79mgODsqsspF1_0NzktvzX-rEsJK6OktdTlp_ISEljkptGWpq5mrBbStw25maHmvdmimZ1iQbJnMrNwxm_m7eHi_NZ2FVSzldR7ePkFM5gql0uAugrkYOpbPix2bSxDrIaD0QCOBHuc2eUfCf40J09ej9eWtPhksxP0LCJxmpDrYQkodHQLFyTwpitX3RhTtOpqum0aOhu6G8rMiHUPtKRhnEHMfljA7HcxNwmfw2SggtGC5B3yX-2tVXH9_bN7qAkfHpApfzOfTBVQd-FqSGA1-XpovyxO3gLDZMtu3YQsOKJGU8FstZx8VknX4e57z_hsbmFAr-FW-KxIwPXWg2GWM3bpU5aABIukmUFDvrIJ_hdVFbdetrgduCslj_u3KKOFRfoS16jBMOikhUNBe0VPCRWpFf_Cv33UaeVAtReAYKOfC7OsY5RtMCWsVaVVlagAYApHJxHVcSV_RzmdZSeSpUYvDCZSGQB7gsnqaKBei6dgE53hADDnJdIuoteEOnf_Cp4IHN_90z.5iqCuk5MimdQ0iCZQN3SnQ/__results___files/__results___37_0.png)

![image.png](https://www.kaggleusercontent.com/kf/80919437/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..DotO8bbfANyqzsfq6vhhpA._AMeyFM5qZCA1nGbqUcQDUcwRp1SJJ-iQHGh5m_ZJNhwmxnI4ivusLZtx9Ozyu5oebEOxCcmlzOCrZzeFmKusxWfWWgxNoGrByMJsvoK4Z730qprl5cuBLGuitPETaRCMc-TTkDpETcqmi55trXbojfiGmOCZ2XL72LLQ24HzGPhMnmcB8k79mgODsqsspF1_0NzktvzX-rEsJK6OktdTlp_ISEljkptGWpq5mrBbStw25maHmvdmimZ1iQbJnMrNwxm_m7eHi_NZ2FVSzldR7ePkFM5gql0uAugrkYOpbPix2bSxDrIaD0QCOBHuc2eUfCf40J09ej9eWtPhksxP0LCJxmpDrYQkodHQLFyTwpitX3RhTtOpqum0aOhu6G8rMiHUPtKRhnEHMfljA7HcxNwmfw2SggtGC5B3yX-2tVXH9_bN7qAkfHpApfzOfTBVQd-FqSGA1-XpovyxO3gLDZMtu3YQsOKJGU8FstZx8VknX4e57z_hsbmFAr-FW-KxIwPXWg2GWM3bpU5aABIukmUFDvrIJ_hdVFbdetrgduCslj_u3KKOFRfoS16jBMOikhUNBe0VPCRWpFf_Cv33UaeVAtReAYKOfC7OsY5RtMCWsVaVVlagAYApHJxHVcSV_RzmdZSeSpUYvDCZSGQB7gsnqaKBei6dgE53hADDnJdIuoteEOnf_Cp4IHN_90z.5iqCuk5MimdQ0iCZQN3SnQ/__results___files/__results___47_0.png)

Sinthetic oversampling techniques were not applied to deal with class imbalance in this case to avoid introducing additional noise to a limited data set.
