# Attention-Guided Neural Networks for Full-Reference and No-Reference Audio-Visual Quality Assessment
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](License)

## Description
ANNAVQA code for the following papers:

- Y. Cao, X. Min, W. Sun and G. Zhai, "Attention-Guided Neural Networks for Full-Reference and No-Reference Audio-Visual Quality Assessment," in IEEE Transactions on Image Processing, vol. 32, pp. 1882-1896, 2023, doi: 10.1109/TIP.2023.3251695.
- Y. Cao, X. Min, W. Sun and G. Zhai, "Deep Neural Networks For Full-Reference And No-Reference Audio-Visual Quality Assessment," 
  2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 1429-1433, doi: 10.1109/ICIP42928.2021.9506408.

## Train models
1. Download the LIVE-SJTU Database

2. Saliency Detection
You should first run sal_position.m in Matlab to get `SJTU_position.mat`. You need to modify the `databasePath` into your save path of the LIVE-SJTU Database.
    ```
    cd Saliency model
    sal_model
    ```

3. Extract video features
    ```
    cd train
    python video_CNNFeatures.py
    ```

4. Extract audio features
    ```
    cd train
    python audio_CNNFeatures.py
    ```
5. Train the FR model
    ```
    cd train
    python ANNAVQA_ref.py
    ```
6. Train the NR model
    ```
    cd train
    python ANNAVQA_noref.py
    ```

