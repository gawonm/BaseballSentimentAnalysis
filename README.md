# ⚾ KBO 야구 하이라이트 댓글 기반 감정 분석 및 요약 프로젝트
오늘날 스포츠는 단순히 경기 결과를 넘어, 팬들과의 감정적 교류가 중심이 되는 경험 중심의 콘텐츠 산업으로 발전하고 있습니다.
특히 팬들의 실시간 반응, 소셜 미디어 댓글등을 통해 팀이나 선수에 대한 감정적 평가가 형성되고 있습니다.

본 프로젝트는 이러한 팬 댓글 데이터를 통해 경기 분석, 경기의 문제점과 핵심 장면을 되짚고, 선수나 팀의 퍼포먼스를 요약 피드백 하는 모델을 구축하고자 합니다. 
팬 반응을 경기 분석에 적극 활용함으로써, 선수와 팀이 실질적인 피드백을 얻고 경기력 향상을 실현할 수 있는 기반을 마련하고자 합니다.

# 📌 연구 배경 및 목적 
### 배경 
최근 스포츠 산업은 단순한 경기 결과 중심에서 벗어나, 팬들과의 감정적 소통을 중시하는 경험 중심 산업으로 변화하고 있다. 특히 유튜브와 같은 SNS 채널에서 팬들은 댓글을 통해 실시간으로 감정과 인식을 표현하며, 이는 선수와 팀의 이미지 형성에 직·간접적인 영향을 미치고 있다.

2025년 KBO 리그는 경기당 평균 관중 15,100명을 기록하며 팬덤이 지속적으로 확장되고 있으며, KBO 유튜브 공식 채널은 누적 2.65억 조회수와 33만 구독자를 보유하며 팬 여론의 주요 공간으로 자리잡고 있다. 
그럼에도 불구하고, 이러한 비정형 감성 데이터를 정량적으로 분석하고 활용하려는 체계적인 연구는 여전히 부족한 실정이다.

<KBO 관중 현황 통계 (2018-2024)>
![kbo.png](https://github.com/gawonm/BaseballSentimentAnalysis/blob/main/kbo.png)

### 목적 
본 연구는 KBO 리그 하이라이트 영상의 댓글 데이터를 수집·정제한 후, 이를 기반으로 감정 분류 및 요약 모델을 개발하여 긍정 및 부정 여론을 자동 분석하는 AI 시스템을 제안한다. 이를 통해 팬 반응을 정량화하여 팀 전략 수립 및 선수 피드백에 실질적으로 활용할 수 있는 데이터 기반의 의사결정 지원 체계를 마련하는 것을 목표로 한다.

# 🔍 주요 목표
- 팬 반응 실시간 분석을 위한 감성 분류 모델 개발
- 핵심 의견 파악을 위한 요약 시스템 구축
- 모델 성능 평가 및 실전 적용 가능성 탐색
  
# 🛠 분석 파이프라인
1. 데이터 수집 및 전처리
2. 감정 이진 분류 라벨링
3. RNN 기반 모델 학습 (LSTM, GRU)
4. Transformers 기반 모델 학습 (KoBERT, RoBERTa, KoELECTRA)
6. TF-IDF 기반 키워드 추출 및 KoBART 기반 댓글 요약
7. 오류 분석 및 클러스터링
8. 결과 시각화 및 적용 방안 논의

# 🧪 데이터셋
학습 데이터 
- **수집 기간**: 2025년 3월 22일 ~ 4월 30일
- **출처**: KBO 리그 유튜브 하이라이트 영상, 티빙스포츠 유튜브 하이라이트 영상 
- **총 댓글 수**: 4,440개
- **라벨링 기준**: 긍정 / 부정 (수작업 라벨링 + 크로스체크)

Test 데이터
- **수집 기간**: 2025년 5월 17일
- **출처**: 삼성 vs 롯데 더블헤더 2차전 경기 하이라이트 영상 
- **총 댓글 수**: 500개

## 댓글 데이터 수집 방법 
1. Youtube API 패키지 설치 
2. Youtube Data API v3 활성화 및 키 발급
3. 클라이언트 객체 초기화 
4. 댓글 수집 함수 정의
5. 수집 대상 경기 리스트 정의
6. 전체 경기 댓글 수집 및 테이블 구성 

# 🏷️ 댓글 라벨링 작업
- 팬 댓글을 긍정 / 부정 두가지 감정으로 분류
- 사전 정의한 기준표에 따라 라벨링 진행
- **자체 제작한 라벨링 도구**를 활용한 라벨링 검수 및 수정 활용
## 댓글 라벨링 도구 (Streamlit 기반 웹 앱)
**💡 아이디어** 

- 자동 라벨링 및 야구 댓글 최적화된 라벨링 모델 개발 어려움 → 수작업 필수
- 라벨링 수작업을 편하고 오류가능성 줄이기 위해 라벨링 도구 개발

💡 **기능 요약**

- 댓글을 하나씩 보여줌
- 버튼 클릭으로 감정 라벨 선택 (긍정 / 부정)
- 라벨링 결과는 바로 CSV로 저장됨
**Step 1: Python + Streamlit 코드**

**Step2 : Streamlit Cloud + GitHub 코드 배포**

Streamlit Cloud는 GitHub에 있는 코드로만 배포 가능

폴더 구성 예시 

labeling-tool/

├── label_app.py        ➡️ 라벨링 Streamlit 코드

├── requirements.txt    ➡️ 패키지 목록

**Step3 : GitHub에 repo 생성 & 코드 업로드**

**Step4 : Streamlit Cloud에서 배포**

라벨링 링크 공유

> https://labeling-tool-xkktcwsxbfcsc6r97cbyuf.streamlit.app/

![label.png](https://github.com/gawonm/BaseballSentimentAnalysis/blob/main/label.png)

# ✏️ EDA (Exploratory Data Analysis)
- 데이터의 감정 레이블 분포를 파악하여 모델 학습 전 클래스 불균형 여부 진단 
![image](https://github.com/user-attachments/assets/707dc3dc-41c6-4170-8fc4-e46e8fd6acf0)

전체 4,440개의 댓글 중,
긍정 레이블(1)이 약 56.5%, 부정 레이블(0)이 약 43.5%로 구성

- 텍스트 길이 분석을 통해 모델 입력 시퀀스 길이 설정 기준 도출
댓글당 평균 단어 수는 약 8.8개, 문자 수는 38자 수준

대부분 댓글은 25단어 이하, 110자 이하에 분포

레이블 간 길이 차이는 크지 않음
![image](https://github.com/user-attachments/assets/cadb1a13-5931-4054-87d9-fce8ebf78a56)

- 댓글 좋아요 수 분포 및 감정 레이블에 따른 반응 차이 분석
![image](https://github.com/user-attachments/assets/812a54c4-6049-43c8-8821-9975d61a7584)

댓글 좋아요 수는 대부분 0~5개 이내로 분포하며, 일부 극단값을 포함해 분포가 right-skewed함 (비대칭)

긍정 레이블(1)의 댓글이 부정(0)보다 좋아요 수 중앙값·평균이 모두 높음

→ 감정 표현과 사용자 반응 사이에 연관 가능성 시사

# 데이터 전처리 
1. 원본 데이터 확인
![image](https://github.com/user-attachments/assets/7ddc0d3a-8d50-478a-b8b0-f8008928bb18)

2. 텍스트 정제 
![image](https://github.com/user-attachments/assets/106303b8-a8b7-453f-a1b0-cb6673d7e237)

3. 형태소 분석 + 불용어 제거
![image](https://github.com/user-attachments/assets/e187a2d7-7456-41ab-867c-e8976bc39673)

4. 시퀀스 변환
![image](https://github.com/user-attachments/assets/b96910c5-3170-4004-ad20-ae2bb191a072)

5. 시퀀스 길이 정규화
![image](https://github.com/user-attachments/assets/e06454ef-368d-4c1d-897b-29c8f6aed353)


# 🧠 사용 모델
- RNN 기반 : LSTM, GRU
- Transformers 기반 : KoBERT, KoRoBERTa,**KoELECTRA**
![image](https://github.com/user-attachments/assets/928ed417-3de5-45e1-88bd-9fdf4abfd634)

## 💬 감정 분류
### 1. RNN 기반 
### 1-1. LSTM 
- LSTM(Long Short-Term Memory)은 RNN의 한 종류
- 기울기 소실 문제를 해결하여 장기 의존성 처리 가능
- 셀 상태(cell state)를 통해 중요한 정보를 장기간 유지
![image](https://github.com/user-attachments/assets/328cf62e-2c31-4999-a991-baf810cb1e90)

- Optimizer : Adam, 안정적 학습을 위한 gradient clipping 적용
- Loss : Binary Crossentropy
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckPoint
- Epochs : 최대 200
- Batch Size : 64
- Validation set : 학습데이터의 20%

### 1-2. GRU 
- LSTM을 발전시킨 구조
- 게이트 수 3 → 2 (Forget Gate 제거)
- 단순화된 구조로 학습 속도 개선
- 파라미터 수 감소 → 연산 효율 ↑
- 유사한 성능 유지 (성능-효율 균형)

  LSTM 층만 GRU 층으로 대체 (LSTM과 동일한 구조 & 학습 조건 사용)

  -> parameter 수가 줄어드는 것 확인
  
LSTM은 적은 epoch에서 최적 모델을 저장했지만, 각 epoch 실행 시간이 더 김
GRU는 더 많은 epoch이 필요했지만, 전체 실행 시간은 유사하거나 더 짧음

최종 성능 

LSTM : 72%

GRU : 70.6%

### 2. Transformers 기반 
### 2-1. KoBERT 
### 2-2. RoBERTa
![image](https://github.com/user-attachments/assets/84a665b6-4559-41ea-bed8-e891dcc3042f)
1. Tokenizer 및 인코딩
2. Dataset, DataLoader 구성
3. 학습 루프 및 평가 수행
4. Epoch별 정확도 및 손실 시각화
5. Confusion Matrix 및 최종 테스트 성능

최종 성능 

KoBERT : 78.8%

RoBERTa : 83.4%

### 2-3. KoELECTRA 
![image](https://github.com/user-attachments/assets/4ebb9df8-ffc2-48ca-8ac1-b4d01619c27c)
01. 라이브러리 및 환경 설정
02. 데이터셋 정의
CommentDataset: 댓글을 토크나이즈해서 input_ids, attention_mask, labels 반환
03. 데이터 전처리 및 분할
학습데이터:검증데이터 = 80:20
04. Tokenization + Dataset 구성
-ElectraTokenizer 사용
-CommentDataset 클래스로 tokenized input 구성
-DataLoader 구성 (shuffle=True for train)
05. 모델 구성
CustomElectra: ElectraModel 위에 dropout + linear layer 추가 → 이진 분류용 로짓 출력

num_labels=1

손실 함수: BCEWithLogitsLoss() 사용 → sigmoid와 함께 이진 분류 설계

06. 하이퍼파라미터 설정
Learning Rate = 2e-5

KoELECTRA는 Pretrained 모델 → 일반적으로 1e-5 ~ 5e-5 사이 권장
2e-5는 논문 기반 실험들과 HuggingFace 권장 값에도 부합

MAX_LEN = 128

대부분의 댓글 128token 이내
성능/효율성 측면에서 균형 잡힌 값으로 선택

PATIENCE = 5 + EarlyStopping

F1 score 기준 5 epoch 연속 성능 미개선 시 학습 중단
불필요한 overfitting 방지 + 시간 단축

07. 모델 학습 (Train Baseline Model)
학습 루프:
1.train 모드로 forward + backward
2.optimizer 업데이트 + gradient clipping
3.검증 시에는 sigmoid 후 THRESHOLD 기반으로 분류
epoch마다 loss/accuracy/F1 저장 및 출력
EarlyStopping 조건 충족하면 중단

08. EarlyStopping
EarlyStopping 구현 (PATIENCE = 5, 기준: F1)
F1-score를 기준으로 삼은 이유: 불균형 감정 데이터에 더 적합

## 최종 선정 모델 : KoELECTRA 
학습 결과 요약 (Epoch별 주요 성능)
![image](https://github.com/user-attachments/assets/7a08ba19-b11c-4e51-a854-1af715abfbab)
![image](https://github.com/user-attachments/assets/abef0ce9-0ae3-4d63-ab70-abdca4d149d5)
![image](https://github.com/user-attachments/assets/58d58405-04d9-4a28-9ad1-f6958a4c1b70)

최종 성능 

KoELECTRA : 85.4%
![image](https://github.com/user-attachments/assets/6fe552f5-0164-4f0e-8d51-9e985345dd29)

### 오답 분석 
![image](https://github.com/user-attachments/assets/b8224644-3018-45b6-8146-bae2117ddb36)

### Test Data 댓글 분석 
![image](https://github.com/user-attachments/assets/fcac52fc-4e50-4b4c-a90b-5435a0e43544)


## 긍/부정 댓글 요약 
## 1. TF-IDF 
![image](https://github.com/user-attachments/assets/19e0c02b-2509-4129-bb86-4450b6b8b4c2)
![image](https://github.com/user-attachments/assets/c732b783-2195-426f-b4b5-27560fd7bdb5)

## 2. KoBART 요약 
![image](https://github.com/user-attachments/assets/084f0f8b-25db-4c20-8098-2e88576fbedd)

# 기대효과 및 의의 

1.  팬 커뮤니케이션의 자동화 기반 마련
- 팬 댓글을 자동으로 감정 분류하고 요약함으로써, 실시간
으로 팬반응을 파악하고 대응할 수 있는 기반 마련
- 수작업 모니터링 대비 시간과 인력 비용을 줄이고, 의사결
정의 객관성 확보

2. 한국어 기반 감정 분석요약 기술의 실제 적용
- KoBERT, KoELECTRA 등 한국어 NLP 모델을
실 데이터에 적용하여 모델별 특성과 성능을 검증
- 고객 리뷰 분석, SNS 대응, 민원 처리 등
다양한 분야로 확장 가능한 기술 기반을 마련
➜ 팬 의견을 데이터화하여 실시간 여론 분석 및 전략
수립에 활용할 수 있는 인공지능 기반 솔루션을 제시

## 개선 방안 
1. 라벨링 데이터 품질 개선
현재 라벨링은 수작업 기준에 따라 진행되어 주관적 편차
가 존재함
향후 활성 학습 기법을 도입하여 정확도와 일관성 향상
2. 모델 다양성 및 하이퍼파라미터 튜닝 확대
세부 하이퍼파라미터 최적화 작업이 제한적이었기에
AutoML이나 GridSearch를 통한 성능 개선 여지가 존
재함
3. 요약 모델 정밀도 개선
4. 피드백 생성 기능의 고도화

## 활용 방안 
1. 스포츠 구단의 팬덤 분석 및 전략 기획
팬 댓글을 실시간으로 분석해 이슈 발생 시 신속한 여론 파
악 및 대응 가능
2. 유튜브·SNS 댓글 자동 분석 시스템 개발
본 프로젝트 파이프라인은 유튜브 댓글 외에도 인스타그
램, 네이버 스포츠 기사 등으로 확장 가능
3. AI 커뮤니케이션 플랫폼과의 연계
요약 결과를 챗봇, 피드백 자동 응답 시스템 등과 연계하면,
사용자의 질문이나 불만에 대해 사전 감정 분석 + 대응 메
시지 자동 생성이 가능

[포스터발표] 
https://github.com/gawonm/BaseballSentimentAnalysis/blob/main/%ED%8F%AC%EC%8A%A4%ED%84%B0%EC%B5%9C%EC%A2%85%EB%B3%B8_5%EC%A1%B0.pdf
![image](https://github.com/user-attachments/assets/d6969c60-d18e-4125-8f13-14b47a7f5302)
