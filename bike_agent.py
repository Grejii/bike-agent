"""
따릉이 실시간 대여 정보 Agent (Gemini 버전, 최소 예제)

준비
  pip install google-genai requests
  환경변수 설정
    SEOUL_API_KEY  : 서울 열린데이터광장 인증키
    GEMINI_API_KEY : Google AI Studio에서 받은 키

실행
  python bike_agent.py
"""
import os
import math
import requests
from google import genai
from google.genai import types

SEOUL_KEY = os.environ["SEOUL_API_KEY"]
# 앞의 모델이 혼잡(503)하거나 쓸 수 없으면 다음 모델로 자동으로 넘어갑니다.
# AI Studio에서 무료로 쓸 수 있는 모델 이름을 확인해 순서를 바꾸거나 추가해도 됩니다.
MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
client = genai.Client()      # GEMINI_API_KEY 환경변수를 자동으로 읽습니다

# LLM은 장소 이름을 좌표로 바꾸지 못하므로, 자주 쓰는 장소만 직접 등록합니다.
# 좌표는 지도 앱에서 꼭 다시 확인하세요.
PLACES = {
    "광운대": (37.6195, 127.0598),
}


# ---------------------------------------------------------------
# 1. 데이터 수집 + 전처리
# ---------------------------------------------------------------
def fetch_stations():
    """bikeList API는 한 번에 최대 1,000건이라 나눠서 받습니다."""
    stations, skipped = [], 0
    start = 1
    while True:
        url = (f"http://openapi.seoul.go.kr:8088/{SEOUL_KEY}"
               f"/json/bikeList/{start}/{start + 999}/")
        data = requests.get(url, timeout=10).json()
        rows = data.get("rentBikeStatus", {}).get("row", [])
        if not rows:
            break
        for r in rows:
            try:  # 문자열로 오는 값을 숫자로 바꾸고, 깨진 행은 건너뜁니다
                stations.append({
                    "name": r["stationName"].strip(),
                    "bikes": int(r["parkingBikeTotCnt"]),
                    "racks": int(r["rackTotCnt"]),
                    "lat": float(r["stationLatitude"]),
                    "lon": float(r["stationLongitude"]),
                })
            except (KeyError, ValueError):
                skipped += 1
        start += 1000
    print(f"[데이터] 대여소 {len(stations)}개 수집, 전처리에서 {skipped}개 제외")
    return stations


def distance_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이 거리(m), 하버사인 공식"""
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


STATIONS = fetch_stations()


# ---------------------------------------------------------------
# 2. Agent가 쓸 도구
#    Gemini는 함수의 이름, 타입 힌트, docstring을 읽고 언제 쓸지 스스로 판단합니다.
#    그래서 docstring이 곧 도구 설명입니다.
# ---------------------------------------------------------------
def search_station(keyword: str) -> dict:
    """대여소 이름에 키워드가 들어간 따릉이 대여소의 실시간 자전거 수를 찾는다.

    Args:
        keyword: 대여소 이름에 들어갈 단어. 예: 석계역
    """
    print(f"  [도구 호출] search_station(keyword={keyword!r})")   # 평가할 때 꼭 보세요
    hits = [s for s in STATIONS if keyword in s["name"]]
    if not hits:
        return {"error": f"'{keyword}'가 들어간 대여소가 없습니다."}
    return {"results": hits[:5]}


def find_nearby(place: str, radius_m: int = 500) -> dict:
    """등록된 장소 주변 반경 안의 따릉이 대여소를 가까운 순으로 찾는다.

    Args:
        place: 등록된 장소 이름. 예: 광운대
        radius_m: 검색 반경(미터). 기본 500
    """
    print(f"  [도구 호출] find_nearby(place={place!r}, radius_m={radius_m})")
    if place not in PLACES:
        return {"error": f"'{place}'의 좌표가 등록되어 있지 않습니다. 등록된 장소: {list(PLACES)}"}
    lat, lon = PLACES[place]
    near = []
    for s in STATIONS:
        d = distance_m(lat, lon, s["lat"], s["lon"])
        if d <= radius_m:
            near.append({**s, "distance_m": round(d)})
    near.sort(key=lambda s: s["distance_m"])
    if not near:
        return {"error": f"{place} 반경 {radius_m}m 안에 대여소가 없습니다."}
    return {"results": near[:5]}


# 응답 기준: 이 문장을 바꿔 가며 정답률이 어떻게 달라지는지 비교해 보세요.
SYSTEM = (
    "너는 서울 따릉이 대여소 안내 Agent다. "
    "반드시 도구 결과에 있는 정보로만 답하고, 결과에 없는 내용은 추측하지 말고 확인할 수 없다고 말한다. "
    "답에는 대여소 이름, 대여 가능 자전거 수, 거리(있으면)를 포함한다."
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[search_station, find_nearby],   # 함수를 그대로 넘기면 SDK가 호출까지 자동으로 처리합니다
)


# ---------------------------------------------------------------
# 3. 질문 -> (Gemini가 도구 호출 -> 결과 확인) 반복 -> 최종 답
# ---------------------------------------------------------------
def ask(question):
    last_error = None
    for model in MODELS:
        try:
            # 질문마다 새 대화를 열어, 앞 질문이 다음 답에 영향을 주지 않게 합니다 (평가할 때 중요)
            chat = client.chats.create(model=model, config=CONFIG)
            res = chat.send_message(question)
            print(f"  [모델] {model}")
            return res.text or "(답변 텍스트가 비어 있습니다. 질문을 조금 더 구체적으로 바꿔 보세요.)"
        except Exception as e:
            print(f"  [모델] {model} 사용 실패 → 다음 모델 시도 ({str(e)[:60]})")
            last_error = e
    raise last_error


if __name__ == "__main__":
    while True:
        q = input("\n질문 (종료: q) > ").strip()
        if q == "q":
            break
        try:
            print(ask(q))
        except Exception as e:   # 서버 혼잡(503), 한도 초과(429) 등이 나도 프로그램이 꺼지지 않게
            print(f"[오류] {e}\n모든 모델이 실패했습니다. 잠시 뒤 다시 질문하거나, MODELS 목록에 AI Studio에서 쓸 수 있는 모델 이름을 추가해 보세요.")