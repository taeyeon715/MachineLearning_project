# 🐾 애완동물 입양 속도 예측 프로젝트

머신러닝을 활용한 애완동물 입양 속도 예측 시스템

## 📋 프로젝트 개요

이 프로젝트는 애완동물의 다양한 특성(나이, 품종, 색상, 건강상태 등)을 분석하여 **입양 속도(AdoptionSpeed)**를 예측하는 머신러닝 모델을 개발합니다.

### 주요 목표
- 애완동물의 특성 데이터 분석 및 전처리
- CatBoost, XGBoost 등 앙상블 모델을 활용한 예측
- 최적의 하이퍼파라미터 탐색을 통한 모델 성능 향상

## 🗂️ 프로젝트 구조

```
MachineLearning_project/
│
├── 기계학습 팀플.ipynb              # 메인 분석 노트북 (CatBoost)
├── 기계학습_Xboost.ipynb            # XGBoost 모델 분석
├── 기계학습_미니언즈_최종발표.pptx  # 최종 발표 자료
│
├── 전처리_withBreeds.csv            # 품종 정보 포함 전처리 데이터
├── BreedName2Num.csv                # 품종 이름-번호 매핑 데이터
└── Metadata.csv                     # 메타데이터
```

## 📊 데이터 특성

### 입력 변수 (Features)
- **기본 정보**: Type(개/고양이), Age(나이), Gender(성별)
- **품종**: Breed1, Breed2
- **외모**: Color1/2/3(색상), MaturitySize(성장 후 크기), FurLength(털 길이)
- **건강**: Vaccinated(예방접종), Dewormed(구충), Sterilized(중성화), Health(건강상태)
- **기타**: Fee(입양비용), State(지역), VideoAmt(동영상 수), PhotoAmt(사진 수)
- **감성 분석**: DesSem(설명 감성 점수), NameSem(이름 감성 점수)

### 타겟 변수 (Target)
- **AdoptionSpeed**: 입양 속도 (분류 문제)

## 🛠️ 기술 스택

### 주요 라이브러리
- **데이터 처리**: `pandas`, `numpy`
- **머신러닝**: `CatBoost`, `XGBoost`, `scikit-learn`
- **시각화**: `matplotlib`, `plotly`

### 사용 모델
※ MLP, SVM, Randaomforest, LightGBM, XGBoost, catBoost 모델 사용 및 5가지 앙상블 모델을 사용했지만 프로젝트 종료 후 로컬 환경에서 코드 유실. -> 프로젝트의 전체적인 흐름과 중간 과정들을 살펴보려면 최종발표 ppt를 보는 것이 참고에 도움 될듯.


1. **CatBoost Classifier**
   - 범주형 변수를 자동으로 처리
   - GridSearchCV를 통한 하이퍼파라미터 튜닝
   
2. **XGBoost**
   - 그래디언트 부스팅 알고리즘
   - 고성능 예측 모델

## 🔍 주요 분석 과정

### 1. 데이터 전처리
-Name과 Descition에 결측치 존재. Name열에 존재하는 결측치는 NoName으로 대체
-입양 수수료인 Fee에 해당하는 변수는 이상치 제거 후 10으로 나눔. 
-color은 검정, 갈색, 크림, 회색, 금색, 하얀색, 노랑색 7가지 색상 존재. oneHot인코딩을 진행. 여러 색이 섞인 강아지나 고양이의 경우 한 인스턴스 안에 모든 색상이 다 적혀있어 최대 4가지 색상이 적힌 강아지가 있었음.
-Quantity는 한번에 입양 가능한 동물 수를 뜻함. 값이 1이 아닌 row는 삭제.
-MetaData 활용 : 'PhotoAmt에 값이 양수로 표시되어 있으면 데이터에 사진도 첨부되어 있음. 사진의 유무가 입양에 가장 큰 영향을 미친다고 생각했기에 이 데이터도 어떻게든 활용하기로 결정.

```python
# 개와 고양이 데이터 분리
dog_df = df[df['Type'] == 1]
cat_df = df[df['Type'] == 2]

# 색상 데이터 One-Hot 인코딩
color_mapping = {1: 'black', 2: 'brown', 3: 'golden', ...}
```

### 2. 특성 엔지니어링
- 색상 데이터 One-Hot 인코딩
- 품종 정보 처리
- 감성 점수 계산
- 상관관계 분석

### 3. 모델 학습 및 튜닝
```python
# 하이퍼파라미터 그리드
param_grid = {
    'iterations': [100, 200, 300, 400],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.3, 0.2, 0.1, 0.01, 0.001]
}

# GridSearchCV로 최적 파라미터 탐색
grid_search = GridSearchCV(model, param_grid, cv=3)
```

### 4. 모델 평가
- 정확도(Accuracy) 측정
- 교차 검증을 통한 성능 평가

## 📈 주요 발견사항

### 상관관계 분석 결과
- **Age와 Sterilized**: -0.273 (나이가 많을수록 중성화율 낮음)
- **Age와 Vaccinated**: -0.146 (나이와 예방접종 음의 상관관계)
- **Type과 Color**: 0.105 (종과 색상의 약한 양의 상관관계)

## 🚀 실행 방법

### 1. 필요한 라이브러리 설치
```bash
pip install pandas numpy catboost xgboost scikit-learn matplotlib plotly
```

### 2. 노트북 실행
```bash
jupyter notebook "기계학습 팀플.ipynb"
```

## 📝 사용 예시

```python
# 데이터 로드
import pandas as pd
data = pd.read_csv('전처리_withBreeds.csv')

# CatBoost 모델 학습
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
```

## 👥 팀 정보

**팀명**: 미니언즈

## 📄 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🔗 참고 자료

- [CatBoost Documentation](https://catboost.ai/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

**마지막 업데이트**: 2024

