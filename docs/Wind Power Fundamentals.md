# Wind Power Fundamentals

### 알아두면 좋은 것  
* 대기경계층(ABL) - 지표면의 영향을 받는 층 (일반적으로 1 ~ 2km 높이)  
    수직혼합과 난류의 영향이 강하다.  

* 바람은 지표면으로부터 높이가 높아질수록 강해진다.  
Wind Profile - 바람의 연직 분포를 나타낸 것  
![alt text](https://www.researchgate.net/profile/Taofiq-Amoloye/publication/325810394/figure/fig2/AS:638474491293696@1529235589501/1The-mean-ABL-wind-velocity-profile-over-different-terrains-25.png)

* 터빈의 구조  
![alt text](https://str.llnl.gov/sites/str/files/2024-04/miller_3_380.jpg)  

* 풍속에 따른 발전 효율  
![alt text](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs00202-023-02005-z/MediaObjects/202_2023_2005_Fig1_HTML.png)


### 바람
풍력 발전이란 공기의 움직임을 포착(capture)해 에너지로 변환하는 것을 말한다.  
따라서 발전량은 공기의 운동에너지에 비례하게 된다.  

바람의 강도는 풍속으로 나타내지고, 운동에너지에 비례하게 된다.  
그리고 이 운동에너지의 원천은 지표면에 가해지는 불균등한(unevenly) 태양복사이다.  
* 공기의 움직임 자체가 이 태양복사로 인한 **대류**에서 만들어진다.  
* (그렇기 때문에 공기의 밀도나 대기안정도와도 연관이 있다.)  

대기과학에서는 대기의 움직임을 4가지 규모로 분리하여 현상을 설명한다.  
* micro, meso, synoptic, planetary  

각 규모는 공간에 대한 규모와 시간에 대한 규모가 있는데, 공간과 시간 규모는 서로 비례관계이다.  
예를 들어 서울의 특정 지역에만 내리는 비는 대한민국 전체적으로 내리는 비에 비해서 금방 그친다. (meso, synoptic의 차이)  
이론적으로는 규모를 분리해서 현상을 설명하게 되지만, 실제 관측하는 바람은 이 규모들이 서로 상호작용한 결과이다. (벡터들의 합이라 생각)  

### 풍력 발전
궁극적으로 풍력 발전에서 가장 궁금한 것은, **지금 불어오는 이 바람에 얼마나 많은 에너지가 존재하는가?** 이다.  
그러기 위해선 먼저 power와 engergy의 차이를 알아야 한다.  
power는 단위 시간당 에너지 (${dE\over dt}$)를 말한다. 
그리고, wind power는 단위 면적당 바람 에너지 흐름의 비율(Flux)을 말한다. ($J/(m^2 \cdot s) \rightarrow W/m^2$)  

$KE = {1\over 2} \cdot m \cdot U^2 \rightarrow P = {1\over 2}\cdot{dm\over dt}\cdot U^2 = {1\over 2}\cdot \rho \cdot A \cdot U^3$  
(kinetic energy -> power)  

${dm\over dt} = \rho \cdot A \cdot U$  
(mass flow rate)

이를 통해 Wind Power는 다음과 같은 내용을 알 수 있다.  
1. Wind Power는 Speed의 영향을 많이 받는다.  
2. Wind Power는 density와 선형적 관계이다. 그러나 밀도의 변화량은 매우 크지 않으므로, 가장 중요한 요인은 아니다.  
3. Wind Power는 Area가 클 수록, 터빈 날개의 길이가 길수록 커진다.  

그리고 위의 수식에서 P는 단위 시간당 에너지 즉, Power를 나타낸 것으로 이를 다시 flux의 형태로 나타내보면  
$WPD = {P\over A} = {1\over 2} \cdot \rho \cdot U^3$  
이렇게 쓸 수 있고 이를 wind power density라고 한다.  
WPD는 터빈 크기에 관계 없이 발전량을 비교하는데 주로 사용된다.  

![alt text](https://www.researchgate.net/publication/319659528/figure/tbl2/AS:668596686557189@1536417280153/International-standards-of-wind-power-generation-classification.png)


