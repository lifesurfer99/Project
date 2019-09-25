import requests
from urllib.parse import quote
 
# 네이버 api call
def call(keyword, start):
    encText = quote(keyword)
    url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&display=100" + "&start=" + str(start)
    result = requests.get(url=url, headers={"X-Naver-Client-Id":"2dgpTHLaxOdSRPPsZpB9",
                                          "X-Naver-Client-Secret":"QTL3SDNRJR"})
    print(result)  # Response [200]
    return result.json()
 
# 1000개의 검색 결과 받아오기
def get1000results(keyword):
    list = []
    for num in range(0,10):
        list = list + call(keyword, num * 100 + 1)['items'] # list 안에 키값이 ’item’인 애들만 넣기
    return list