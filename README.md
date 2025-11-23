# 🐾 애완동물 입양 속도 예측 프로젝트

머신러닝을 활용한 애완동물 입양 속도 예측 시스템

## 📋 프로젝트 개요

이 프로젝트는 애완동물의 다양한 특성(나이, 품종, 색상, 건강상태 등)을 분석하여 **입양 속도(AdoptionSpeed)**를 예측하는 머신러닝 모델을 개발합니다.

### 주요 목표

* 애완동물의 특성 데이터 분석 및 전처리
* RandomForest, XGBoost 등 앙상블 모델을 활용한 예측
* 최적의 하이퍼파라미터 탐색을 통한 모델 성능 향상
* 강아지와 고양이 데이터를 분리하여 각각 최적화된 모델 구축

## 🗂️ 프로젝트 구조

```
MachineLearning_project/
│
├── 기계학습 팀플.ipynb              # 메인 분석 노트북 (CatBoost)
├── 기계학습_Xboost.ipynb            # XGBoost 모델 분석
├── 입양속도예측_앙상블_최종 (2).html # 앙상블 모델 최종 결과 (RandomForest + XGBoost)
├── 기계학습_미니언즈_최종발표.pptx  # 최종 발표 자료
│
├── 전처리_withBreeds.csv            # 품종 정보 포함 전처리 데이터
├── BreedName2Num.csv                # 품종 이름-번호 매핑 데이터
└── Metadata.csv                     # 메타데이터
```

## 📊 데이터 특성

### 입력 변수 (Features)

* **기본 정보**: Type(개/고양이), Age(나이), Gender(성별)
* **품종**: Breed1, Breed2, BreedName, BreedNum, Mixed(믹스 여부)
* **외모**: Color1/2/3(색상), MaturitySize(성장 후 크기), FurLength(털 길이)
* **건강**: Vaccinated(예방접종), Dewormed(구충), Sterilized(중성화), Health(건강상태)
* **기타**: Fee(입양비용), State(지역), VideoAmt(동영상 수), PhotoAmt(사진 수)
* **감성 분석**: Description_SCORE(설명 감성 점수)
* **이미지 메타데이터**: Object_SIZE, Object_COLOR(H/S/V), Image_WIDTH, Image_HEIGHT

### 타겟 변수 (Target)

* **AdoptionSpeed**: 입양 속도 (0-4 분류 문제)
  - 0: 같은 날 입양
  - 1: 1주일 이내 입양
  - 2: 1개월 이내 입양
  - 3: 3개월 이내 입양
  - 4: 입양되지 않음

## 🛠️ 기술 스택

### 주요 라이브러리

* **데이터 처리**: `pandas`, `numpy`
* **머신러닝**: `RandomForest`, `XGBoost`, `CatBoost`, `LightGBM`, `scikit-learn`
* **시각화**: `matplotlib`, `plotly`

### 사용 모델

1. **RandomForest Classifier**
   * 최적 하이퍼파라미터 적용
   * GridSearchCV를 통한 파라미터 튜닝

2. **XGBoost**
   * 그래디언트 부스팅 알고리즘
   * 고성능 예측 모델

3. **앙상블 모델 (RandomForest + XGBoost)**
   * 두 모델의 예측 결과를 평균하여 최종 예측
   * 더 안정적이고 정확한 예측 성능

## 🔍 주요 분석 과정

### 1. 데이터 전처리

* Name과 Description 결측치 처리 (NoName으로 대체)
* 입양 수수료(Fee) 이상치 제거 및 정규화 (10으로 나눔)
* Color 데이터 One-Hot 인코딩 (7가지 색상: 검정, 갈색, 크림, 회색, 금색, 하얀색, 노랑색)
* Quantity 필터링 (한 번에 1마리만 입양 가능한 경우로 제한)
* Metadata 활용: 사진 메타데이터 추출 및 특성 생성

### 2. 특성 엔지니어링

* 색상 데이터 One-Hot 인코딩
* 품종 정보 처리 및 믹스 여부 판단
* 감성 점수 계산
* 이미지 메타데이터 활용

### 3. 모델 학습 및 튜닝

```python
# 최적 하이퍼파라미터
best_params = {
    'bootstrap': True,
    'criterion': 'entropy',
    'max_depth': 44,
    'min_samples_leaf': 2,
    'min_samples_split': 8,
    'n_estimators': 696
}

# RandomForest 모델 생성
rf_model = RandomForestClassifier(random_state=42, **best_params)
rf_model.fit(X_train, y_train)

# XGBoost 모델 생성
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# 앙상블 예측
ensemble_preds = ((rf_model.predict(X_valid) + xgb_model.predict(X_valid)) / 2).round().astype(int)
```

### 4. 모델 평가

* 정확도(Accuracy) 측정
* 정밀도(Precision) 측정 (macro average)
* 재현율(Recall) 측정 (macro average)
* F1 스코어 측정 (macro average)

## 📈 앙상블 모델 성능 결과

### 전체 데이터 (개 + 고양이)

* **정확도**: 65.89%
* **정밀도**: 66.43%
* **재현율**: 64.97%
* **F1 스코어**: 64.69%

### 강아지 데이터

* **정확도**: 66.04%
* **정밀도**: 66.86%
* **재현율**: 66.48%
* **F1 스코어**: 65.94%

### 고양이 데이터

* **정확도**: 65.66%
* **정밀도**: 66.17%
* **재현율**: 64.73%
* **F1 스코어**: 64.33%

> **참고**: 강아지와 고양이를 분리하여 모델을 학습시킨 결과, 강아지 데이터에서 가장 높은 성능을 보였습니다.

## 🚀 실행 방법

### 1. 필요한 라이브러리 설치

```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm matplotlib plotly
```

### 2. 노트북 실행

```bash
jupyter notebook "기계학습 팀플.ipynb"
```

또는 앙상블 모델 결과 확인:

```bash
# 브라우저에서 입양속도예측_앙상블_최종 (2).html 파일 열기
```

## 📝 사용 예시

```python
# 데이터 로드
import pandas as pd
data = pd.read_csv('ML_Teamproject_Minions_Data.csv', index_col=0)

# 특성과 정답 분리
features = data.drop('AdoptionSpeed', axis=1)
labels = data['AdoptionSpeed']

# 학습/검증 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42)

# RandomForest 모델 학습
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42, **best_params)
rf_model.fit(X_train, y_train)

# XGBoost 모델 학습
from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# 앙상블 예측
rf_preds = rf_model.predict(X_valid)
xgb_preds = xgb_model.predict(X_valid)
ensemble_preds = ((rf_preds + xgb_preds) / 2).round().astype(int)

# 평가
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_valid, ensemble_preds)
print(f"앙상블 모델 정확도: {accuracy}")
```

## 💡 주요 발견사항

### 앙상블 기법의 효과

* RandomForest와 XGBoost의 예측을 결합하여 단일 모델보다 안정적인 성능 달성
* 두 모델의 강점을 활용하여 과적합 방지

### 데이터 분리의 중요성

* 강아지와 고양이 데이터를 분리하여 학습시킨 결과, 각 동물 유형에 최적화된 모델 구축 가능
* 강아지 데이터에서 가장 높은 성능 기록

### 특성 중요도

* 사진 메타데이터(Object_SIZE, Object_COLOR 등)가 입양 속도 예측에 중요한 역할
* 품종 정보와 색상 정보도 유의미한 영향

## 👥 팀 정보

**팀명**: 미니언즈

## 📄 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

## 🔗 참고 자료

* [RandomForest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [CatBoost Documentation](https://catboost.ai/docs/)
* [scikit-learn Documentation](https://scikit-learn.org/)

---

**마지막 업데이트**: 2024

