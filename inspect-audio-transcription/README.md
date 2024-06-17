# inspect-audio-transcription

## 개요
전사 프로젝트 검수시 도움을 줄 수 있는 프로젝트임.

## TODO
1. tts wav 파일 확보
2. wav파일을 libsora 패키지로 실수형 음성신호로 변환
3. 음성신호 전처리 (초반 무음 부분 삭제)
4. tts wav와 원본 wav의 size를 동일하게 맞춤. (최대한 원본 wav는 훼손되지 않도록 tts를 조정함)
5. feature를 tts wav로 target을 원본 wav로 DNN 모델 학습을 진행함.
5. tts wav에 대한 prediction data는 원본 wav와 유사한 형태로 변하게 됨.
  * 이때 음성 전사가 올바르게 되었을 경우 원본 wav와 transcription이 유사한 형태를 보일것으로 예상. 유사도가 높게 나옴.
  * 음성 전사가 올바르게 되지 않을 경우 prediction data 또한 원본 wav와 큰 차이를 보일것으로 예상. 유사도가 낮게 나옴.
6. prediction data와 원본 wav간의 유사도를 0~1의 실수로 표현함.

