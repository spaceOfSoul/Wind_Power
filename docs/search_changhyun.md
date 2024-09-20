# 간단한 딥러닝 테스트
    
    https://github.com/spaceOfSoul/Analysis_WindTurbine
    

- Date/Time이 일자별로 정규화 안됨. 월별로, 계절별로 다른 특징을 만들어주어야 할듯.
- NMAE 아직 도출 안함.
- 일간 평균 MSE : 0.2 내외
- 사용하는 feature : Date/Time,LV ActivePower (kW),Wind Speed (m/s),Theoretical_Power_Curve (KWh),Wind Direction (°)

딥러닝 모델 기준 feature를 그대로 넣고 하면 예측결과가 그리 좋지 않음. gradient 특성은 잘 따라가는 것 같은데 고점이 있는 느낌.

추가적인 post-processing이 필요할 수 있음.

# kaggle
### by Mohamed Taher. Wind Turbine Power Prediction [R² = 98.19%]. Kaggle Codes
---
- XGBRegressor 모델 정확도  (RSqaure = 98.2% & RMSE = 0.13)라고 함.
- 특정 각도에서 높은 풍속이 나타나기 때문에 Yaw 시스템의 존재는 의미가 없음
    - Yaw 는 회전 각도를 말하는 듯함.
- 사용하는 feature : 낮/밤, 계절, 온도, 날짜, 시간, 주, 월, Effective Theoretical Power
    - Effective Theoretical Power : 터빈의 최대 전력 (이론상)
- 낮/밤의 여름 및 겨울 자전축 기울기를 적용하여 높은 정확도의 낮/밤 시간.
- 온도 특성 생성은 다양한 위치에서 온도가 크게 다르기 때문에 좋은 방법이 아니지만, 이즈미르 지역에 풍력 터빈이 많이 있으므로 모델에 도움이 될 수 있음. 터빈의 정확한 위치를 알기 전까지는 이점이 있을 것이며, 낮/밤의 경우 모든 위치에서 차이가 크지 않음.

- 이외에도 데이터 전처리에 주목할 부분이 많음.
  
  - 터빈이 전력을 생산하기 위한 최소 풍속(컷인 속도) 이상인 행만 남기고 데이터를 필터링



# 딥러닝을 이용한 풍력발전량 예측
### Choi, J.-G., & Choi, H.-S. (2021). Prediction of Wind Power Generation using Deep Learning. Journal of the KIECS, 16(2), 329-338.
---
- Bidirectional LSTM 과 CNN을 결합한 하이브리드 모델 사용. 논문에서는 단순 MLP와의 비교를 수행함.
- RMSE(Root Mean Square Error)와 MAPE(Mean Absolute Percentage Error) 지표에서 각각 8.706, 5.487.
- 비교 대상이 단순 MLP 모델이라, 정말 좋은지는 더 봐야할 듯 함.
  

왜 CNN을 앞에 달았는지 모르겠음. feature가 적은데 굳이 conv가 필요한가 싶음.

attention 메커니즘 적용이 나을 듯. feature가 적은 상황에서 포커스맞출 특징맵을 원한다면 attention이 cnn 레이어보다는 더 나은 방법일 것 같음.
  
  =>  attention 달고 시도 ㄱㄱ

# 2019 재생e 발전량 예측 경진대회(풍력) - 경주풍력발전
- 공기 밀도를 계산하여 feature로 사용. (대기압, 대기온도)
- 고도차에 따른 풍속 보정


# 