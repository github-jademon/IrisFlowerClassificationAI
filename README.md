﻿# IrisFlowerClassificationAI (붓꽃 품종 분류)

![IrisFlowerClassificationAI](https://github.com/github-jademon/IrisFlowerClassificationAI/assets/79764169/98d868b6-1f1c-46bd-b42d-a2f7f773a7c8)

이 프로젝트는 붓꽃의 품종을 예측하는 머신러닝 모델입니다. 해당 AI는 주어진 붓꽃의 특성 정보(sepal length, sepal width, petal length, petal width)를 통해 'setosa' 'versicolor' 'virginica'으로 품종을 분류합니다.

## 목차

1. [프로젝트 설명](#프로젝트-설명)
2. [데이터셋](#데이터셋)
3. [실행환경](#실행환경)
4. [프로젝트 설치 및 실행 방법](#프로젝트-설치-및-실행-방법)
5. [프로젝트 사용 방법](#프로젝트-사용-방법)

## 프로젝트 설명

이 프로젝트는 Python과 scikit-learn을 사용하여 붓꽃 품종 분류 AI를 개발한 것입니다. 붓꽃 품종 분류 AI 모델은 LogisticRegression(로지스틱 회귀•분류)와 PolynomialFeatures을 활용하여 'setosa' 'versicolor' 'virginica'으로 다중 분류를 수행합니다.

## 데이터셋

- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)

## 실행환경

- [Cuda (release 11.8, V11.8.89)](https://developer.nvidia.com/)
- [Anaconda (conda 23.7.1)](https://www.anaconda.com/)
- [Python (Python 3.11.4)](https://www.python.org/)

## 프로젝트 설치 및 실행 방법

1. 먼저, Python 및 필요한 라이브러리를 설치합니다.

   ```bash
   pip install scikit-learn numpy
   ```

2. `main.py` 스크립트를 실행하여 모델을 훈련합니다.

## 프로젝트 사용 방법

`main.py` 스크립트를 실행하여 붓꽃 품종 분류를 진행합니다.

붓꽃의 특성(꽃받침과 꽃입의 가로, 세로길이)을 입력하면 AI 모델이 해당 입력을 바탕으로 붓꽃 품종을 분류하여 출력합니다.

- 품종은 "setosa", "versicolor", "virginica"으로 분류됩니다.

<img src="https://github.com/github-jademon/IrisFlowerClassificationAI/assets/79764169/74e16bc3-935e-4dbb-a9ef-56cd24e58dd3" alt="IrisFlowerClassificationAI" width="300" />

이 프로젝트에 관한 문의 사항이나 버그 리포트는 [j2python@gmail.com]로 보내주세요.
