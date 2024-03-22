한글명: 노이즈 데이터에 따른 애플 주가 예측 정확도 분석 (LSTM & Sentence Sentiment Classification)  
영문명: Enhancing Apple Inc. (AAPL) Stock Price Prediction through LSTM and Sentiment Analysis: A Focus on Noise Data Integration


* 1단계: 주가 데이터 수집 및 전처리  
* 2단계: noise 데이터 수집 및 전처리 << 웹 크롤링 사용  
* 3단계: LSTM 모델 생성
  * 주가 데이터만 학습한 모델
  * 주가 데이터 + Noise데이터(index 지수) 학습한 모델
  * 주가 데이터 + Noise데이터(뉴스 기사) 학습한 모델
  * 주가 데이터 + Noise데이터(index 지수, 뉴스 기사) 학습한 모델
* 4단계: 평가
  
