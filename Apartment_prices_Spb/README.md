# Apartment Evaluation Models

This repository contains code for training apartment pricing models based on gradient boosting.

Data source: https://www.kaggle.com/mrdaniilak/russia-real-estate-20182021/

Original implementation: https://www.kaggle.com/ekaterinadranitsyna/russian-housing-evaluation-model

**Models used:** XGBoost, LGBM, CatBoost

**Techniques applied:** data cleaning, filtering outliers, feature engineering

Data set contains listings for Russian Housing from 2018 till 2021 across different regions, a total of 5,477,006 samples with 12 features and a target value ("price").

**Features:**
- 'date' - date the listing was published
- 'time' - exact time the listing was published
- 'geo_lat' - geographical coordinate of the property
- 'geo_lon' - geographical coordinate of the property
- 'region' - numerically encoded geographical area
- 'building_type' - numerically encoded type of the building where the apartment is located
- 'level' - floor the apartment is located on
- 'levels' - total number of storeys in the building
- 'rooms' - number of rooms in the apartment (-1 stands for studios with open-space layout)
- 'area' - total floor area of the apartment in sq. meters
- 'kitchen_area' - kitchen area in sq. meters
- 'object_type' - apartment type, where 1 stands for secondary real estate market, 11 - new building

In data cleaning phase I enforce correct data types, remove excessive information like exact time the listing was published, remove or correct price and area outliers. For filtering purposes I limit the floor area to a range between 20 sq.m and 200 sq.m and price to a range between 1.5 mln to 50 mln rubles, which covers most of the housing market. For model training I use a subset of data representing Saint Petersburg region.

In feature engineering step I introduce two temporal features (year and month of the listing publishing) and two numeric ratios (ratio of floor area to a number of rooms and ratio of floor to total number of floors in the buiding). The rationale behind this is that more spacious rooms usually indicate that the apartment belongs to a more prestigious and upper-tier residential area. Apartments on upper floors are more expensive but the price also depends on the total number of floors: 5th floor in a 25-storey building is not the same at 5th floor in a 5-storey building.

![image.png](https://www.kaggleusercontent.com/kf/81089532/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..f_jmNdfrbFanNxpu7wx3hw.DwhSsLt0ksYY02NfPknNCex5hDxw7lrOZBfNYQu2kO1DMs_UCbzfdA7wPv67-9WugX2K8JtqvFyKovky1V3Vo0z7K8VZ3DG4aJ_8f6HeeS48PS316gdwusggKFNs5IkjQBgiQAmDvRybwzme8JgfQEMIYSMC11VudjSqUNVe9DxSupKllgidG7W5fpRJ6_4olUh5S-_bs2V03NQ-HOubrNvhK3GKbAlhlW7bLrt1Oo7jIHT9n3_WJI1-8Rma3WmiCBCfWJiDjJCQaYsot_daxwDMemJlJDBLklDg3EPvgyZIxJX5d3nM1P-opW_yQmMEhfnFz_yrcIi4T3uBusCBSDpJiFd20En0Bb9n2p1O573-zzEAt9afd4D-BgOy7QvWve3etK-yav9FZvcaCP9-CHrKh6-9l02WmI83JTroMctBoKMrs-7jzBqQEmuvAeeTmONjPBBfg8Q6vHE7NS_tMYk2jlXpZUzaL_IZ6C5gg7gRQlqjAfHrcEpXVqfuBBGMnvl6mrPGLfbvctseRqc3tMuw-waKWHypWHzxR9MQL0AO-6Fvgoac1C1isPk3g3itxwiSpxR9TFhyRmahXKnKREvp2rHCiWFDwZ8nIMRTtjr0zsLCPGv1bTY3PJTfx9bqAQGejajFT7wg-isHHVHroem7NTxQ_X_Dmgy1fRC7vOQwHo-3dtsU17jghVVjvJEB._cwy4MEWNDGQ_zY7P5Y0EA/__results___files/__results___27_0.png)

**Standard deviation of price** in the selected subset of data representing Saint Petersburg housing market is about **5.6 mln rubles**.

**Models performance (5-fold validation RMSE):**
- CatBoost - 1.736 mln rubles
- XGBoost - 1.796 mln rubles
- LGMM - 1.944 mln rubles

Models assigned different importances to input features, which implies that a combination of several models could make predictions more reliable.

![image.png](https://www.kaggleusercontent.com/kf/81089532/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..f_jmNdfrbFanNxpu7wx3hw.DwhSsLt0ksYY02NfPknNCex5hDxw7lrOZBfNYQu2kO1DMs_UCbzfdA7wPv67-9WugX2K8JtqvFyKovky1V3Vo0z7K8VZ3DG4aJ_8f6HeeS48PS316gdwusggKFNs5IkjQBgiQAmDvRybwzme8JgfQEMIYSMC11VudjSqUNVe9DxSupKllgidG7W5fpRJ6_4olUh5S-_bs2V03NQ-HOubrNvhK3GKbAlhlW7bLrt1Oo7jIHT9n3_WJI1-8Rma3WmiCBCfWJiDjJCQaYsot_daxwDMemJlJDBLklDg3EPvgyZIxJX5d3nM1P-opW_yQmMEhfnFz_yrcIi4T3uBusCBSDpJiFd20En0Bb9n2p1O573-zzEAt9afd4D-BgOy7QvWve3etK-yav9FZvcaCP9-CHrKh6-9l02WmI83JTroMctBoKMrs-7jzBqQEmuvAeeTmONjPBBfg8Q6vHE7NS_tMYk2jlXpZUzaL_IZ6C5gg7gRQlqjAfHrcEpXVqfuBBGMnvl6mrPGLfbvctseRqc3tMuw-waKWHypWHzxR9MQL0AO-6Fvgoac1C1isPk3g3itxwiSpxR9TFhyRmahXKnKREvp2rHCiWFDwZ8nIMRTtjr0zsLCPGv1bTY3PJTfx9bqAQGejajFT7wg-isHHVHroem7NTxQ_X_Dmgy1fRC7vOQwHo-3dtsU17jghVVjvJEB._cwy4MEWNDGQ_zY7P5Y0EA/__results___files/__results___33_0.png)

![image.png](https://www.kaggleusercontent.com/kf/81089532/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..f_jmNdfrbFanNxpu7wx3hw.DwhSsLt0ksYY02NfPknNCex5hDxw7lrOZBfNYQu2kO1DMs_UCbzfdA7wPv67-9WugX2K8JtqvFyKovky1V3Vo0z7K8VZ3DG4aJ_8f6HeeS48PS316gdwusggKFNs5IkjQBgiQAmDvRybwzme8JgfQEMIYSMC11VudjSqUNVe9DxSupKllgidG7W5fpRJ6_4olUh5S-_bs2V03NQ-HOubrNvhK3GKbAlhlW7bLrt1Oo7jIHT9n3_WJI1-8Rma3WmiCBCfWJiDjJCQaYsot_daxwDMemJlJDBLklDg3EPvgyZIxJX5d3nM1P-opW_yQmMEhfnFz_yrcIi4T3uBusCBSDpJiFd20En0Bb9n2p1O573-zzEAt9afd4D-BgOy7QvWve3etK-yav9FZvcaCP9-CHrKh6-9l02WmI83JTroMctBoKMrs-7jzBqQEmuvAeeTmONjPBBfg8Q6vHE7NS_tMYk2jlXpZUzaL_IZ6C5gg7gRQlqjAfHrcEpXVqfuBBGMnvl6mrPGLfbvctseRqc3tMuw-waKWHypWHzxR9MQL0AO-6Fvgoac1C1isPk3g3itxwiSpxR9TFhyRmahXKnKREvp2rHCiWFDwZ8nIMRTtjr0zsLCPGv1bTY3PJTfx9bqAQGejajFT7wg-isHHVHroem7NTxQ_X_Dmgy1fRC7vOQwHo-3dtsU17jghVVjvJEB._cwy4MEWNDGQ_zY7P5Y0EA/__results___files/__results___36_0.png)

![image.png](https://www.kaggleusercontent.com/kf/81089532/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..f_jmNdfrbFanNxpu7wx3hw.DwhSsLt0ksYY02NfPknNCex5hDxw7lrOZBfNYQu2kO1DMs_UCbzfdA7wPv67-9WugX2K8JtqvFyKovky1V3Vo0z7K8VZ3DG4aJ_8f6HeeS48PS316gdwusggKFNs5IkjQBgiQAmDvRybwzme8JgfQEMIYSMC11VudjSqUNVe9DxSupKllgidG7W5fpRJ6_4olUh5S-_bs2V03NQ-HOubrNvhK3GKbAlhlW7bLrt1Oo7jIHT9n3_WJI1-8Rma3WmiCBCfWJiDjJCQaYsot_daxwDMemJlJDBLklDg3EPvgyZIxJX5d3nM1P-opW_yQmMEhfnFz_yrcIi4T3uBusCBSDpJiFd20En0Bb9n2p1O573-zzEAt9afd4D-BgOy7QvWve3etK-yav9FZvcaCP9-CHrKh6-9l02WmI83JTroMctBoKMrs-7jzBqQEmuvAeeTmONjPBBfg8Q6vHE7NS_tMYk2jlXpZUzaL_IZ6C5gg7gRQlqjAfHrcEpXVqfuBBGMnvl6mrPGLfbvctseRqc3tMuw-waKWHypWHzxR9MQL0AO-6Fvgoac1C1isPk3g3itxwiSpxR9TFhyRmahXKnKREvp2rHCiWFDwZ8nIMRTtjr0zsLCPGv1bTY3PJTfx9bqAQGejajFT7wg-isHHVHroem7NTxQ_X_Dmgy1fRC7vOQwHo-3dtsU17jghVVjvJEB._cwy4MEWNDGQ_zY7P5Y0EA/__results___files/__results___39_0.png)

There is still room for improvement taking into account that median price in the city is about 6 mln rubles. However without additional features making these models significantly more accurate would be difficult. A number of important features are missing in this data set:
- Condition: similar apartments would be priced differently, if one of them is being sold fully furnished in excellent condition and the other is being sold without finishing.
- Ceiling height affects the price.
- Balconies and terraces increase the price compared to similar apartments without such amenities.
- Additional unique properties usually mentioned in the description of the apartment, like chimneys or underground parking spaces, affect the price.
