import requests
import urllib.parse
from dotenv import load_dotenv
import os

load_dotenv()

service_key = os.getenv('SERVICE_KEY')

url = "https://apis.data.go.kr/B552070/wthrService/oamsFile24"
decoded_service_key = urllib.parse.unquote(service_key)
headers = {'Content-Type': 'application/json'}

params = {
    'serviceKey': decoded_service_key,
    'pageNo': 1,
    'numOfRows': 10,
    'apiType': 'json',
    'bizplcNm': '당진',
    'msrmtYmd': '2023-09-30 23:00'
}

response = requests.get(url, params=params, headers=headers, verify=False)

if response.status_code == 200:
    data = response.json()
    print("API 호출 성공")
    print(data) # 받은 데이터
else:
    print(f"API 호출 실패. 상태 코드: {response.status_code}, 메시지: {response.text}")
