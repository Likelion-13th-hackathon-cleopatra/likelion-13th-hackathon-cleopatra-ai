import os, json, glob, csv, asyncio, collections, argparse, re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
import sys
# 파일 상단 어딘가(모듈 전역)에 추가
from datetime import datetime
from itertools import count
REPORT_ID_SEQ = count(1)  # 1,2,3... 자동 증가

sys.stdout.reconfigure(encoding='utf-8')


from schemas import (
    DocIn, DocOut, Step1Keywords, PlatformSummary
)
import os, json

def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)

def save_json(path: str, data):
    ensure_outputs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------------
# 0) 환경설정 & 클라이언트
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 비어 있습니다. .env 또는 환경변수 설정을 확인하세요.")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL_STEP1   = "gpt-4.1-mini"
MODEL_EXPLAIN = "gpt-4.1-mini"
MODEL_REPORT  = "gpt-4.1-mini"
CONCURRENCY   = 8

# 필드 표준화 맵
FIELD_ALIASES = {
    "감성 키워드": "감성_키워드",
}

STEP1_FIELDS = ["주체","행동","수단","시간","동기","감성_키워드","경쟁요소"]

# 본문/ID 키 후보
TEXT_KEYS = ["text", "content", "review", "body"]
ID_KEYS   = ["doc_id", "id"]

# -------------------------
# 1) 로컬 로더
# -------------------------
def _guess_text(obj: Dict[str, Any]) -> Optional[str]:
    for k in TEXT_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k]
    return None

def _guess_id(obj: Dict[str, Any]) -> Optional[str]:
    for k in ID_KEYS:
        if k in obj and isinstance(obj[k], (str, int)):
            return str(obj[k])
    return None

def _platform_from_filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def load_docs_by_platform_jsons(data_dir: str) -> List[DocIn]:
    docs: List[DocIn] = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    for p in paths:
        platform = _platform_from_filename(p)
        with open(p, "r", encoding="utf-8") as f:
            arr = json.load(f)

        if not isinstance(arr, list):
            raise ValueError(f"{p} 의 최상위 구조는 리스트여야 합니다.")

        for i, item in enumerate(arr, 1):
            if not isinstance(item, dict):
                continue
            text = _guess_text(item)
            if not text:
                continue
            doc_id = _guess_id(item) or f"{platform}_{i:04d}"
            docs.append(DocIn(doc_id=doc_id, platform=platform, text=text))
    print(f"[i] Loaded {len(docs)} docs from {data_dir}")
    return docs

# -------------------------
# 2) 파이프라인: 1A 키워드 추출
# -------------------------
def _normalize_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """필드명 통일"""
    normalized = {}
    for k, v in data.items():
        key_std = FIELD_ALIASES.get(k, k)
        normalized[key_std] = v
    # 누락 필드는 빈 문자열로
    for f in STEP1_FIELDS:
        if f not in normalized:
            normalized[f] = ""
    return normalized

import re, json

def _clean_json_str(txt: str) -> str:
    if not txt:
        return "{}"   # ✅ 빈 문자열일 때도 안전하게 "{}" 반환

    raw = txt

    def try_load(s: str) -> str:
        json.loads(s)
        return s

    # 1) 그대로 시도
    try:
        return try_load(txt.strip())
    except:
        pass

    # 2) 코드펜스 제거
    s = txt.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()


    # 4) BOM 제거
    s = s.lstrip("\ufeff")

    # 5) 따옴표 통일
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

    # 6) 최초 '{' ~ 최후 '}' 절단
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and l < r:
        cand = s[l:r+1]
        try:
            return try_load(cand)
        except:
            s = cand

    # 9) 최종 시도
    try:
        return try_load(s)
    except Exception:
        return raw   # ✅ 여기서 원문 그대로 반환

# -------------------------
# OpenAI 응답 안전 추출 헬퍼
# -------------------------
def _safe_output_text(resp) -> str:
    """
    OpenAI responses.create() 응답에서 텍스트/JSON을 안전하게 추출.
    - output_text
    - output[*].content[*].text
    - output[*].content[*].json  ← ✨ 추가
    - content[*].text / content[*].json
    """
    # 1) SDK가 제공하는 표준 속성
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    def _scan_content_list(lst):
        # content 리스트에서 text/json 둘 다 탐색
        for c in lst:
            if not isinstance(c, dict):
                continue
            # (a) text 우선
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            # (b) json/object 타입 지원
            j = c.get("json")
            if j is not None:
                try:
                    return json.dumps(j, ensure_ascii=False)
                except Exception:
                    # json 직렬화 실패 시 문자열 캐스팅
                    return str(j)

    try:
        # 2) responses API: resp.output / resp.outputs
        out = getattr(resp, "output", None) or getattr(resp, "outputs", None)
        if isinstance(out, list):
            for block in out:
                content = block.get("content") if isinstance(block, dict) else None
                if isinstance(content, list):
                    hit = _scan_content_list(content)
                    if isinstance(hit, str) and hit.strip():
                        return hit

        # 3) 일부 구현체: resp.content
        content = getattr(resp, "content", None)
        if isinstance(content, list):
            hit = _scan_content_list(content)
            if isinstance(hit, str) and hit.strip():
                return hit
    except Exception:
        pass

    # 4) 최후 fallback: 문자열화
    return (str(resp) or "").strip()






async def _extract_keywords(doc: DocIn) -> DocOut:
    system = (
    "너는 시장/상권 문서에서 카테고리 키워드를 뽑는 분석가다.\n"
    "입력 본문(text)만 근거로 하며, 임의 추론/상상 금지. 사실이 없으면 빈 문자열 \"\"을 넣는다.\n"
    "출력은 반드시 JSON 객체 1개이며, 다음 7개 키만 포함한다(추가/누락 금지):\n"
    "[\"주체\",\"행동\",\"수단\",\"시간\",\"동기\",\"감성_키워드\",\"경쟁요소\"]\n\n"
    "각 키의 의미/추출 기준:\n"
    "- 주체(Who): 누가 이 지역을 이용하는지. 예) \"대학생\",\"직장인\",\"가족\"\n"
    "  · 본문에서 명시된 사용자군/인구집단 명사로 1개 선택.\n"
    "- 행동(What): 무엇을 위해 지역을 찾는지. 예) \"야식 주문\",\"카페 방문\",\"장보기\"\n"
    "  · 본문 내 활동/행위 표현 1개. 동사/명사구 가능.\n"
    "- 수단(How): 어떤 방법으로 행동을 실행하는지. 예) \"배달 앱\",\"직거래\",\"SNS 공유\"\n"
    "  · 채널/도구/방식 1개.\n"
    "- 시간(When): 언제 이용이 집중되는지. 예) \"주말\",\"저녁\",\"과제 기간\"\n"
    "  · 시간대/요일/시즌/상황 중 1개.\n"
    "- 동기(Why): 왜 이 행동을 하는지. 예) \"스트레스 해소\",\"공부 집중\",\"문화 체험\"\n"
    "  · 목적/이유를 1개.\n"
    "- 감성_키워드: 리뷰/댓글의 긍·부정 반응 수식어. 예) \"맛있다\",\"시끄럽다\",\"친절하다\",\"불편하다\"\n"
    "  · 감정/평가 형용사·형용동사 1개.\n"
    "- 경쟁요소: 창업 적합도에 영향을 주는 주변 환경/비용 요소. 예) \"상가 밀집\",\"주차 난이도\",\"임대료\"\n"
    "  · 비용/접근성/집적도/규모 등 1개.\n\n"
    "형식/제약:\n"
    "- \"행동\"은 반드시 2~4자의 명사 카테고리로만 작성한다. 예) \"식사\",\"포장\",\"면류\",\"사시미\",\"초밥\",\"덮밥\"\n"
    "- \"행동\"에 동명사/동사(먹기, 마시기, 맛보기, 주문하기, 방문, 이용 등)나 조사/부사/수식어를 쓰지 마라.\n"
    "- \"행동\"에 메뉴 + 동사 조합(예: 면 먹기, 사시미 맛보기, 초밥 주문)은 금지하고, 명사형만 남겨라(예: 면류, 사시미, 초밥).\n"
    "- \"행동\"이 비어야 할 정도로 애매하면 \"식사\"로 둔다.\n"
    "- 각 필드는 최대 12자. 12자를 넘으면 의미가 깨지지 않게 앞부분 기준으로 자연스럽게 줄여라(예: \"노원역 1번출구\"→\"노원역 1번\").\n"
    "- 값이 여러 개면 본문에서 가장 자주/강하게 언급된 1개만 고른다(동률 시 더 구체적인 표현 우선).\n"
    "- 중복 금지: 한 필드에 쓴 표현을 다른 필드에 재사용하지 않는다.\n"
    "- 지시문/해설/코드펜스(```json 등) 금지. JSON만 출력.\n"
    "- 값에는 줄바꿈/따옴표/이모지/해시태그/불릿을 넣지 말 것.\n"
    "플랫폼명(NAVER_BLOG, NAVER_PLACE, YOUTUBE 등)이나 '플랫폼'이라는 단어는 절대 추출하지 마라.\n"
    "- 본문에 근거가 없으면 \"\"(빈 문자열)을 넣는다.\n\n"
    "- '맛있다', '좋다' 같은 뻔한 단어는 절대 추출하지 마라.\n"
    "- '방문', '이용' 같은 일반적이고 의미 없는 행동 단어는 절대 추출하지 마라.\n"
    "검증(스스로 점검 후 출력):\n"
    "1) 모든 키가 존재하는지 확인.\n"
    "2) 각 값이 12자 이내인지 확인(길면 줄임).\n"
    "3) 본문에 근거 없는 생성이 없는지 확인.\n\n"
    "4) 취소선 절대 포함 금지\n"
    "출력 예시(형식만 참고):\n"
    "{\"주체\":\"대학생\",\"행동\":\"카페 방문\",\"수단\":\"배달 앱\",\"시간\":\"주말\",\"동기\":\"공부 집중\",\"감성_키워드\":\"친절하다\",\"경쟁요소\":\"임대료\"}\n"
     )

    user_payload = [
        {"type": "input_text", "text": doc.text}
    ]
    resp = await client.responses.create(
        model=MODEL_STEP1,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.2
    )
    raw_text = _safe_output_text(resp)
    cleaned = _clean_json_str(raw_text)
    try:
        data = json.loads(cleaned)
    except Exception as e:
        print(f"[!] JSON 파싱 실패: {e} | 원문: {raw_text[:100]}")
        data = {}
    data = _normalize_fields(data)
    return DocOut(doc_id=doc.doc_id, platform=doc.platform, keywords=Step1Keywords(**data))

async def run_step1_extract_all(docs: List[DocIn]) -> List[DocOut]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def run(d: DocIn):
        async with sem:
            return await _extract_keywords(d)
    tasks = [run(d) for d in docs]
    return [await coro for coro in asyncio.as_completed(tasks)]

#플랫폼별 통합 탑5 키워드
def build_platform_top5_overall(step1_docs: List[DocOut], fields=None, n=3) -> List[Dict[str, Any]]:
    """
    플랫폼마다 STEP1의 모든 필드(주체/행동/수단/시간/동기/감성_키워드/경쟁요소)를 통합하여
    가장 자주 등장한 값 Top-N을 계산합니다.
    반환: [{platform, top_overall: [키워드...], counts: {키워드:빈도}, evidence_map: {키워드:[doc_id...]}}]
    """
    if fields is None:
        fields = ["주체","행동","수단","시간","동기","감성_키워드","경쟁요소"]

    by_pf: Dict[str, List[DocOut]] = collections.defaultdict(list)
    for row in step1_docs:
        by_pf[row.platform].append(row)

    def _top_n(values: List[str], n=3) -> List[str]:
        c = collections.Counter([v for v in values if v])
        return [w for w, _ in c.most_common(n)]

    out: List[Dict[str, Any]] = []
    for pf, rows in by_pf.items():
        # 플랫폼 한 묶음에서 모든 필드 값 통합 수집
        merged_values: List[str] = []
        evidence_map: Dict[str, List[str]] = {}
        for r in rows:
            kw = r.keywords.model_dump()
            for f in fields:
                val = (kw.get(f) or "").strip()
                if not val:
                    continue
                merged_values.append(val)
                evidence_map.setdefault(val, []).append(r.doc_id)

        # Top-N 추출
        top_overall = _top_n(merged_values, n=n)

        # 카운트도 같이 제공(필요하면 UI/디버깅에 유용)
        counts = dict(collections.Counter(merged_values))

        out.append({
            "platform": pf,
            "top_overall": top_overall,
            "counts": counts,                 # 선택적으로 사용
            "evidence_map": evidence_map      # 키워드별 근거 doc_id
        })
    return out

# -------------------------
# 3) 플랫폼 집계 & 해석
# -------------------------
def _top_n(values: List[str], n=3) -> List[str]:
    c = collections.Counter([v for v in values if v])
    return [w for w, _ in c.most_common(n)]

def build_platform_summaries(step1_docs: List[DocOut], n_top=3) -> List[Dict[str, Any]]:
    by_pf = collections.defaultdict(list)
    for row in step1_docs:
        by_pf[row.platform].append(row)

    summaries = []
    for pf, rows in by_pf.items():
        field_values = {f: [] for f in STEP1_FIELDS}
        evidence_map = {}
        for r in rows:
            kw = r.keywords.model_dump()
            for f in STEP1_FIELDS:
                val = kw.get(f) or ""
                field_values[f].append(val)
                if val:
                    evidence_map.setdefault(val, []).append(r.doc_id)
        top_keywords = {f: _top_n(field_values[f], n_top) for f in STEP1_FIELDS}
        summaries.append({"platform": pf, "top_keywords": top_keywords, "evidence_map": evidence_map})
    return summaries

async def add_platform_explanations(platform_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async def run(item):
        pf = item["platform"]
        sys = (
            "너는 플랫폼별 특성을 1~2문장으로 요약하는 분석가다. "
            "주어진 상위 키워드들만 근거로 한 줄 해석을 작성하라."
            "취소선 절대 사용 금지."
            "- 반드시 제공된 top_keywords만 근거로 작성해.\n"
            "너는 시장/상권 보고서를 작성하는 분석가다.\n"
            "결과는 'AI 해설' 카드처럼 짧고 설명적인 요약문으로 작성한다.\n\n"
            "스타일 가이드:\n"
            "- 문체: 간결하고 자연스러운 구어체 보고서 문장\n"
            "- 길이: 2~3문장 (한 단락 요약)\n"
            "- 어투: '~이에요.', '~있어요.', '~자리잡았어요.' 처럼 정중하지만 딱딱하지 않은 어투\n"
            "- 어투: '~이에요.', '~있어요.', '~찾아요.' 처럼 가볍고 자연스럽게\n"
            "- 금지: '이는', '긍정적인 경험', '자리잡았음을 보여줘요' 같은 어색하거나 번역투 느낌의 표현 사용 금지\n"
            "- 불필요한 수식어나 추상적인 단어는 쓰지 않는다\n"
            "- 추상적이거나 어색한 표현은 금지.\n"
            "- 마케팅/광고 같은 어투(예: '최고다', '선택지', '인기 있다')는 쓰지 않는다.\n"
            "- 데이터나 패턴을 먼저 설명하고, 그 의미를 덧붙인다\n"
            "- 불필요한 감정/추측/수사는 금지\n"
            "- 플랫폼명(NAVER_BLOG, NAVER_PLACE, YOUTUBE 등)이나 '플랫폼'이라는 단어는 절대 언급하지 말 것\n"
            "- 출력에는 해설 문장만 포함. '설명:', '분석:' 같은 접두사는 쓰지 않는다.\n\n"
        )
        user_json = json.dumps({"platform": pf, "top_keywords": item["top_keywords"]}, ensure_ascii=False)
        resp = await client.responses.create(
            model=MODEL_EXPLAIN,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": sys}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_json}]}
            ],
            temperature=0.2
        )
        text = _safe_output_text(resp)
        return {**item, "explanation": text}

    return [await coro for coro in asyncio.as_completed([run(s) for s in platform_summaries])]

# -------------------------
# 4) 보고서 생성 (API 명세 반영 버전, 새 응답 포맷 래핑)
# -------------------------
async def generate_report_v2(
    area: str,
    category: str,
    platform_data: List[Dict[str, Any]],     # data 배열 (platform, platform_keyword, platform_description)
    population: Dict[str, Any],              # 전체 population 객체
    price: Dict[str, Any],                   # 전체 price 객체
    income_consumption: Dict[str, Any],      # 전체 income_consumption 객체
):
    """
    네가 준 API 명세의 모든 필드를 그대로 user_context에 넣고,
    LLM에게 '그 데이터만' 근거로 보고서를 쓰게 해요.
    - 말투: -해요/-돼요/-이에요
    - 굵게(**) 강조 사용
    - offers/kpis/improvements: head + body 구조
    - improvements.body는 '단어/구' 나열
    """

    system_prompt = (
        "당신은 데이터 분석 기반 상권·창업 전략 보고서를 작성하는 전문가예요.\n"
        "입력은 지역(area), 업종(category), 플랫폼별 요약(data), 인구(population), 부동산 시세(price), "
        "소득 및 지출 구조(income_consumption) 정보를 포함해요. 새로운 키워드나 해석을 생성하지 말고 "
        "반드시 주어진 내용만 서술해요. 임의 추정, 보간, 재계산을 금지해요.\n"
        "\n"
        "[데이터 신뢰/반영 규칙]\n"
        "- 모든 문장은 입력 JSON의 값만 근거로 작성해요. 존재하지 않는 값은 절대 만들어내지 않아요.\n"
        "- 키 오타가 있을 수 있어요. 다음 별칭은 같은 것으로 간주해요: "
        "  paercent→percent, spanding_toal→spending_total. 그래도 값을 찾지 못하면 사용하지 않아요.\n"
        "- 특정 키/값이 없거나 비어 있으면 해당 문구는 '~~'로 남겨요(예: summary.body_3, population.gender 등).\n"
        "- 수치 표기는 입력값을 그대로 사용하고, 비율/숫자를 새로 계산하거나 추정하지 않아요.\n"
        "- 성별 문장(gender)은 population.gender.percent의 male_percent, female_percent '원시값'만 사용해요. "
        "둘 중 하나라도 없으면 '~~'로 남겨요.\n"
        "- 연령대 문장(age)은 population.ages.percent의 값만 사용해요. 없으면 '~~'로 남겨요.\n"
        "- 거래량(volume)은 price.trading_volume의 값만 사용해요. 없으면 '~~'로 남겨요.\n"
        "- 소득/지출 문장(income_consumption)은 income.monthly_income_average, income.income_class_code, "
        "consumption.percent의 상위 항목만 사용해요. 없으면 '~~'로 남겨요.\n"
        "- 창업 전략적 해석(적합 업장 규모, 비용 민감도, 타이밍 등)은 허용하지만, 반드시 위 입력값을 근거로 하고 "
        "추가 수치 생성이나 재계산 없이 서술로만 표현해요.\n"
        "\n"
        "[출력 형식]\n"
        "- 반드시 JSON 한 덩어리만 출력해요. 코드펜스와 여분 텍스트는 포함하지 않아요.\n"
        "- 출력 키는 정확히 다음만 사용해요(철자/구조 변경 금지):\n"
        "  description_summary{head, line_1, line_2, line_3},\n"
        "  description_population{age, gender},\n"
        "  description_price{value_average, value_pyeong, volume},\n"
        "  description_income_consumption,\n"
        "  description_strategy{review{head, body[]}, kpi{head, body[]}, improvements{head, body[]}}.\n"
        "\n"
        "[작성 지시]\n"
        "1) description_summary: 전체 내용을 3~4문장으로 요약해요.\n"
        " description_summary의 head, body 각 항목에는 **반드시 한 개 이상의 핵심 구/키워드, 수치**를 **굵게** 표시한다. 단, 핵심어만 굵게 처리한다.\n"
        "   - head: 입력 JSON의 population.total_resident, population.ages.percent, income.income_class_code 값을 기반으로 작성해요.\n"
        "      이 항목에는 **반드시 한 개 이상의 핵심 구/키워드, 수치**를 **굵게** 표시한다. 단, 항목 전체를 전부 굵게 만들지는 말고 필요한 핵심어만 굵게 처리한다.\n"
        "     · 총 인구수(population.total_resident)는 '실거주 인구 약 N명' 형식으로 표현해요.\n"
        "     · 연령대(population.ages.percent)는 비율이 가장 높은 1~2개 구간을 뽑아 '20대,30대 비중이 높은'처럼 반드시 쉼표 구분으로만 표기해요.\n"
        "     · 연령대는 반드시 **어린 연령대부터 큰 연령대 순서**(예: '20대,30대', '30,40대')로만 나열해요.\n"
        "       (절대 '20~40대', '20-40대' 같은 표기는 쓰지 말아요.)\n"
        "     · 소득 수준은 income.income_class_code 숫자가 클수록 소득이 높은 지역으로 간주해요.\n"
        "       예: code가 높으면 '소득 수준이 높은 편', 낮으면 '소득 수준이 낮은 편'이라고만 표현해요.\n"
        "     · 표현 예시: \"{지역명}은 실거주 인구 약 **5만 명**으로, **중위소득 수준의 30,40대 비중이 높은 지역**이에요.\"\n"
        "     · 단, 입력 값이 없으면 해당 부분은 생략하고, 인구·연령·소득 모두 없으면 '~~'로 남겨요.\n"
        "     · '빈곤하다' 등 자극적이고 예민할 수 있는 단어는 절대 쓰지 말아요.\n" 
        "\n"
        "   - line_1: population.ages.percent를 근거로 연령대별 소비 전략을 요약해요.\n"
        "       · 이 항목에는 **반드시 한 개 이상의 핵심 구/키워드, 수치**를 **굵게** 표시한다. 단, 항목 전체를 전부 굵게 만들지는 말고 필요한 핵심어만 굵게 처리한다.\n"
        "       · 특정 연령대(예: 20대,30대 / 40,50대)가 주도하는 경우 → 그 계층 중심 전략을 언급해요.\n"
        "     · 연령대는 반드시 **어린 연령대부터 큰 연령대 순서**(예: '20대,30대', '30,40대')로만 나열해요.\n"
        "       · 두 연령대가 균형 잡힌 경우 → 세대 혼합형 소비 전략을 언급해요.\n"
        "       · 반드시 '20대,30대'처럼 쉼표 구분으로만 작성하고, '20~40대' 같은 범위 표기는 절대 쓰지 마요.\n"
        "       · 문장은 간결하고 창업자가 바로 이해할 수 있게 작성해요.\n"
        "       · 예: \"20대,30대 비중이 높아 트렌드 반응형 아이템이 유리해요. 40대 이상은 일정 비중이라 세대 혼합 전략도 필요해요.\"\n"
        "\n"
        "   - line_2: 부동산 데이터( price.* )를 **창업 적합성 관점**으로 해석해요.\n"
        "       · 이 항목에는 **반드시 한 개 이상의 핵심 구/키워드, 수치**를 **굵게** 표시한다. 단, 항목 전체를 전부 굵게 만들지는 말고 필요한 핵심어만 굵게 처리한다.\n"
        "       · value_average: 소형은 **초기 임대 부담과 빠른 회전**, 대형은 **가족·단체 수요 확보** 같은 특징을 연결해 설명해요.\n"
        "       · 두 유형 중 **count 값이 더 큰 쪽**을 중심으로 작성하고, "
        "       · value_pyeong: 평당가 수준을 근거로 **객단가 전략**이나 **회전율 관리** 시사점을 덧붙여요.\n"
        "       · volume: 최근 거래 흐름을 근거로 **유입 활발**, **안정적**, **협상 여지** 중 하나로 표현해요.\n"
        "       구체적인 수치 인용은 하지 말고, 맥락과 경향만 자연스럽게 설명해요. 값이 모두 없으면 '~~'로 둬요.\n"
        "\n"
        "   - line_3: data[].platform_keyword와 platform_description을 근거로 "
        "       · 이 항목에는 **반드시 한 개 이상의 핵심 구/키워드, 수치**를 **굵게** 표시한다. 단, 항목 전체를 전부 굵게 만들지는 말고 필요한 핵심어만 굵게 처리한다.\n"
        "       · 방문 시간대/행동 패턴/경쟁요소를 요약해요.\n"
        "       · 방문 시간대(예: 저녁, 주말, 점심),\n"
        "       · 행동 패턴(예: 산책 중 방문, 가족 단위 외식, 반려견 동반 등),\n"
        "       · 경쟁요소(예: 카페 밀집, 임대료, 접근성) 중 입력에 있는 것만 선택해\n"
        "       한 문장으로 창업 시 고려해야 할 **생활 패턴/경쟁 환경 요약**을 작성해요.\n"
        "       (입력에 없는 키워드는 새로 만들지 말고, 중복 없이 자연스럽게 구성해요.)\n"
        "\n"
        "2) description_population: {age, gender} 구조로 작성해요.\n"
        "  - 절대로 굵게(**) 표시하지 않는다(별표 금지). "
        "   - 연령대 표기는 반드시 쉼표 구분으로만 써라. 예) '20대,30대', '40,50대'\n"
        "     · 연령대는 반드시 **어린 연령대부터 큰 연령대 순서**(예: '20대,30대', '30,40대')로만 나열해요.\n"
        "   - '20~40대', '20-40대' 같은 표기는 절대 쓰지 마라.\n"
        "   - 출력 시 절대로 마크다운 취소선(`~~내용~~`)을 쓰지 않아요.\n"
        "   - age: population.ages.percent에서 비중이 큰 연령대를 근거로 한 문장 + 소비전략을 함께 적어요.\n"
        "     예: \"20대,30대 비중이 X% 내외예요. 활동성이 높아 회전율 관리와 저녁 특화 전략이 좋아요.\" percent가 없다면 '~~'.\n"
        "   - gender: population.gender.percent의 남성/여성 비율을 비교해 긴 문장으로 작성해요.\n"
        "     · 두 값이 모두 있으면 다음 규칙을 따르세요.\n"
        "      1) 차이가 3%p 이상이면, 비율이 큰 성별을 **굵게** 퍼센트와 함께 제시하고,\n"
        "          해당 성별의 일반적인 소비 패턴과 창업 전략을 연결해요.\n"
        "          - 여성이 많으면: 카페·디저트, 뷰티·라이프스타일, 소규모 모임 수요와 연결.\n"
        "          - 남성이 많으면: 식사·주류, 스포츠·여가, 실속 있는 메뉴와 연결.\n"
        "       2) 차이가 3%p 미만이면, '남녀 비중이 비슷해요'라고 서술하고,\n"
        "          성별 중립형 전략(예: 취향 선택권 확대, 가족 단위/커플 동반 적합)으로 연결해요.\n"
        "     · 값이 하나만 있으면 해당 성별만 언급하고 간단한 전략만 제시해요.\n"
        "     · 영어(male/female) 표기는 금지하고 반드시 **남성/여성**으로만 표기해요.\n"
        "     · 괄호 안 원시값은 표기하지 않아요(괄호 자체 금지).\n"
        "     · 문맥이 어색하지 않게 자연스럽게 쓰고, 과장된 표현을 피하세요.\n"
        "     · 값이 없으면 '~~'로 남겨요.\n"

        "3) description_price: {value_average, value_pyeong, volume} 구조로 작성해요.\n"
        " 절대로 굵게(**) 표시하지 않는다(별표 금지). "
        "- 연령대 표현 시 무조건 ','을 이용해 '20대,30대'처럼 작성해요.\n"
        "     · 연령대는 반드시 **어린 연령대부터 큰 연령대 순서**(예: '20대,30대', '30,40대')로만 나열해요.\n"
        "- 출력 시 절대로 마크다운 취소선(`~~내용~~`)을 절대 사용하지 않아요.\n"
        "   - value_average: 반드시 price.big{big_average,big_middle,big_count}와 "
        "     price.small{small_average,small_middle,small_count} 값을 사용해요.\n"
        "     · 모든 수치는 '약 …만원' 형식으로 표기해요(원→만원 환산, 반올림).\n"
        "     · 소형/대형 업장별 평균값·중위값·데이터 건수를 언급하되 단순 비교로 끝내지 말고, "
        "창업자가 실제로 활용할 수 있는 전략적 의미를 제시해요.\n"
        "       예: '소형 업장은 평균 약 6,000만원, 중위값 약 5,800만원(13건)으로 "
        "초기 진입 부담이 상대적으로 낮아 소규모 테스트 창업이나 회전율 중심 업종에 적합해요. "
        "반면 대형 업장은 평균 약 8,000만원(거래 15건)으로 비용 부담이 크지만, "
        "가족·단체 중심 업종을 안정적으로 운영할 수 있는 기반을 마련해줘요. "
        "거래 데이터 건수가 충분하므로 시세 신뢰도가 높아요.'\n"
        "     · 해석은 반드시 '창업자가 어떤 전략을 취해야 하는가'에 초점을 맞추고, "
        "그냥 '높다/낮다'로만 끝내지 말아요.\n"
        "     · 값이 없으면 생략하고, 전부 없으면 '~~'로 남겨요.\n"

        "   - value_pyeong: 반드시 price.price_per_meter, price.price_per_pyeong 값을 사용해 "
        "단위 면적당 시세를 제시하고, 창업 전략가 관점에서 자세히 2-3줄로 해석해요.\n"
        "     · 제시할 때는 모두 '약 …만원' 형식으로 표기해요(원→만원 환산, 반올림).\n"
        "     · 서울시 평균값은 ㎡당 약 1212만원, 평당 약 4000만원으로 고정해 비교해야 해요.\n"
        "     · 비교 시 단순히 '높다/낮다'로 끝내지 말고, 창업자가 어떤 전략을 취해야 하는지까지 연결해요.\n"
        "       · 평균보다 높으면: '서울 평균 대비 높은 수준이에요. "
        "임대·운영 부담이 크므로 **객단가 전략 강화**, **고급화·차별화**가 필요해요.'\n"
        "       · 평균보다 낮으면: '서울 평균 대비 낮은 수준이에요. "
        "임대 부담이 상대적으로 적어 **소규모 창업**, **빠른 회전 전략**으로 기회가 있어요.'\n"
        "     · 서울 평균과 유사한 경우에는: '서울 평균과 비슷한 수준이에요. "
        "안정적인 비용 구조를 바탕으로 **입지와 서비스 차별화**에 집중하는 게 좋아요.'\n"
        "     · 값이 없으면 '~~'로 남겨요.\n"

        
        "   - volume: price.trading_volume의 **최근 4개 분기 값만** 근거로 거래 흐름을 판단해요.\n"
        "       먼저 한 문장으로 추세를 요약해요(라벨은 **증가 / 감소 / 정체 / 혼조 / 회복** 중 하나를 사용).\n"
        "       이어서 창업 관점에서 해석을 3~4문장 정도의 줄글로 설명해요. 숫자를 나열하지 말고, "
        "추세가 창업 의사결정에 어떤 의미가 있는지 자연스럽게 이어주세요.\n"
        "       예를 들어, '최근 거래량이 감소 흐름이에요. 임차 조건을 유리하게 협상할 기회가 될 수 있고, "
        "입지를 신중히 고르면 안정적인 운영 기반을 만들 수 있어요. 다만 보행 동선과 가시성이 좋은 위치를 선택하는 것이 중요해요. "
        "또한 고정비 부담을 관리하며 장기적인 안정성을 확보하는 전략이 필요해요.'처럼 3~4문장으로 작성해요.\n"
        "       분기명은 '2024년 3분기'처럼 말로만 언급하고, **구체 숫자는 쓰지 않아요**. "
        "필요하다면 '소폭', '뚜렷이' 같은 서술형 표현을 사용해요.\n"

        "\n"
        "4) description_income_consumption:\n"
        "- 반드시 한국어만 사용하고, 굵게(**)·취소선(~~)은 절대 쓰지 마세요.\n"
        "- 2~3문장으로 간결하게, 일상적인 톤으로 요약하세요. 마케팅 용어 대신 '먹거리/즐길 거리/생활 편의'처럼 자연스러운 표현을 쓰세요.\n"
        "- 아래 입력만 근거로 사용하고, 없으면 그 부분은 자연스럽게 생략하세요.\n"
        "  • income.monthly_income_average  (정수, 원 단위)\n"
        "    • income.income_class_code       (문자열 숫자; 숫자가 클수록 소득 수준이 높은 편)\n"
        "  • consumption.percent            (딕셔너리: 항목→비율)  없으면 consumption.expend(항목→금액)을 대신 사용\n"
        "  • spending_total / spanding_toal 같은 오탈자 키는 동일 의미로 취급\n"
        "\n"
        "[표현 규칙]\n"
        "① 금액 표기는 모두 'xx만원'으로 통일하세요(원→만원 환산, 반올림; 숫자 외 단위·기호 추가 금지).\n"
        "② 소득 구간은 income_class_code만 근거로 '소득 수준이 높은 편/낮은 편'처럼만 표현하고, 코드 숫자 자체는 쓰지 마세요.\n"
        "③ monthly_income_average가 있으면 '월평균 소득은 약 xx만원 수준이에요'를 한 번만 언급하세요.\n"
        "④ 소비 성향은 consumption.percent에서 비중 상위 2~3개 항목을 고르고, percent가 없으면 consumption.expend의 금액 상위 2~3개로 대체하세요.\n"
        "⑤ 영문 키는 한국어로 치환하세요(영문 표기 사용 금지).\n"
        "   - eating_out→음식, leisure_culture→여가/문화, clothing_footwear→의류, living_goods→생활용품,\n"
        "     medical→의료비, transport→교통, education→교육,\n"
        "     food→식료품, entertainment→오락, other→기타\n"
        "⑥ 문맥이 어색하지 않게 자연스럽게 쓰고, 자극적인 표현은 피하세요.\n"
        "\n"
        "[작성 가이드]\n"
        "- 첫 문장: 소득 수준 요약(예: '소득 수준이 높은 편이에요').\n"
        "- 이어서 월평균 소득을 'xx만원'으로 제시(있을 때만).\n"
        "- 마지막 문장: 상위 2~3개 소비 항목을 한국어로 묶어 생활 맥락으로 설명(예: '외식과 여가·문화 지출이 두드러져 먹거리와 즐길 거리에 대한 수요가 이어져요').\n"
    
        "5) description_strategy:\n"
        "- 연령대 표기는 반드시 쉼표 구분으로만 써라. 예) '20대,30대' ( '20~40대', '20-40대' 금지 )\n"
        "     · 연령대는 반드시 **어린 연령대부터 큰 연령대 순서**(예: '20대,30대', '30,40대')로만 나열해요.\n"
        "- 출력에서 마크다운 취소선(~~)은 절대 쓰지 말아라.\n"
        "   - review: 반드시 객체(Object)로 출력한다.\n"
        "     {\"head\":\"핵심 요약 문장\",\"body\":[\"실행 아이디어\", ...]} 구조를 지킨다.\n"
        "     이 항목의 head는 절대로 굵게(**) 표시하지 않는다(별표 금지). "
        "head에는 리뷰 데이터를 바탕으로 창업자가 참고할 수 있는 서비스 제안의 목적이나 기대 효과를 한 줄로 자연스럽게 요약해라.\n"
        "     body는 항상 문장 배열로 작성하며, 각 항목은 번호 없이 자연스럽게 서술형으로 작성한다. "
        "각 문장 안에는 **반드시 한 개 이상의 핵심 구/키워드**를 **굵게** 표시한다. "
        "항목 전체를 전부 굵게 만들지는 말고 필요한 핵심어만 강조한다.\n"
        "     내용은 반드시 data[].platform_keyword 와 platform_description 을 근거로만 작성한다.\n"
        "     예시: {\"head\":\"리뷰 기반으로 고객 반응을 높일 수 있는 실행 전략이에요.\", "
        "\"body\":[\"**SNS 이벤트**를 통해 방문 후기 공유를 유도할 수 있어요.\", "
        "\"예약 고객을 대상으로 **프라이빗 룸 할인**이나 서비스 업그레이드를 제공해요.\", "
        "\"인기 메뉴를 중심으로 **패키지 상품**을 개발하면 가성비 효과를 줄 수 있어요.\", "
        "\"**배달 플랫폼 연계 서비스**를 강화해 고객 접점을 넓혀요.\"]}\n"
        "  - kpi: 반드시 객체(Object)로 출력한다.\n"
        "    {\"head\":\"문장\",\"body\":[\"지표\", ...]} 구조를 지킨다.\n"
        "    head는 절대로 굵게(**) 표시하지 않는다(별표 금지). 아래 문구를 반드시 포함한다: "
        "    '상기 지표를 기반으로 마케팅, 운영, 서비스 개선 전략을 수립하는 게 좋아요. 주 단위 점검을 통해 전략 실행 속도를 빠르게 최적화할 수 있어요.'\n"
        "    body의 **모든 항목은 항목 전체를 굵게(**)** 표시**한다. (예: \"**야간 방문 고객 수**\")\n"
        "    내용 근거는 data[].platform_keyword / platform_description 에서만 가져온다.\n"
        "  - improvements: 반드시 객체(Object)로 출력한다.\n"
        "    {\"head\":\"문장\",\"body\":[\"개선/조치\", ...]} 구조를 지킨다.\n"
        "    head는 절대로 굵게(**) 표시하지 않는다(별표 금지). 다음 문구를 반드시 포함한다: "
        "    '정기적으로 개선점을 관리하는 게 비용 누수와 고객 불만을 사전에 차단하는 데 좋아요.'\n"
        "    body의 **모든 항목은 항목 전체를 굵게(**)** 표시**한다. (예: \"**대기 동선 분리 및 좌석 안내 고도화**\")\n"
        "    불편 요소와 권장 조치는 data[].platform_description에서만 도출한다.\n"
        "[스타일/톤]\n"
        "- 금지: '이는', '긍정적인 경험', '자리잡았음을 보여줘요' 같은 어색하거나 번역투 느낌의 표현 사용 금지\n"
        "- 불필요한 수식어나 추상적인 단어는 쓰지 않는다\n"
        "- 추상적이거나 어색한 표현은 금지.\n"
        "- 마케팅/광고 같은 어투(예: '최고다', '선택지', '인기 있다')는 쓰지 않는다.\n"
        "- 모든 문장은 '-해요/-돼요/-이에요' 어투로 작성해요.\n"
        "- 전체 분량은 800~1200자 내외를 권장해요.\n"
        "- 출력 시 절대로 마크다운 취소선(`~~내용~~`)을 절대 사용하지 않아요.\n" 
        "- 값이 없을 때는 \"~~\" (placeholder 문자열)만 그대로 남겨요.\n"
        "- 즉, '취소선' 서식은 금지하고, 단순한 문자열 \"~~\"만 허용해요.\n"
        "\n"
        "[검증 체크리스트]\n"
        "- 출력 JSON은 지정한 키만 포함해요. 누락/오탈자/추가 키가 있으면 즉시 스스로 수정해요.\n"
        "- 각 필드에 근거 키가 없을 경우 해당 값은 '~~'로 남겨요. 임의 추정/보간/재계산은 절대 하지 않아요.\n"
        "- age/gender 문장은 각각 population.ages.percent, population.gender.percent 값만 사용했는지 확인해요.\n"
        "- price(value_average/value_pyeong/volume) 문장은 지정한 키(big/small/price_per_meter/price_per_pyeong/"
        "trading_volume)의 값만 사용했는지 확인하고, 해석은 서술형으로만 해요(새 수치 생성 금지).\n"
    )

    # 유저 입력 context
    user_context = {
        "area": area,
        "category": category,
        "data": platform_data,
        "population": population,
        "price": price,
        "income_consumption": income_consumption,
    }

    # LLM 호출
    resp = await client.responses.create(
        model=MODEL_REPORT,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(user_context, ensure_ascii=False)}]},
        ],
        temperature=0.2,
    )
    raw_text = _safe_output_text(resp)
    cleaned = _clean_json_str(raw_text)

    # 파싱 성공 여부에 따라 status
    try:
        llm_out = json.loads(cleaned)  # description_* 구조
        status = "success"
    except Exception:
        return {
            "status": "fail",
            "data": None
        }

    # 메타 파생: primary/secondary (category에서)
    if isinstance(category, str):
        parts = category.split()
        primary = parts[0] if len(parts) >= 1 else "외식업"
        secondary = parts[1] if len(parts) >= 2 else "일식"
    else:
        primary, secondary = "외식업", "일식"

    # report_id / created_at / sub_neighborhood (district, neighborhood 제거)
    report_id = next(REPORT_ID_SEQ)
    created_at = datetime.utcnow().isoformat()
    sub_neighborhood = area or "공릉 1동"  # 요청: area 대신 sub_neighborhood만 출력. 없으면 기본값.

    # description_* 안전 추출
    ds = llm_out.get("description_summary", {}) or {}
    dp = llm_out.get("description_population", {}) or {}
    dprice = llm_out.get("description_price", {}) or {}
    dic = llm_out.get("description_income_consumption", "") or ""
    dst = llm_out.get("description_strategy", {}) or {}

    # head/body 정규화
    def _section(obj):
        head = (obj or {}).get("head") if isinstance(obj, dict) else ""
        body = (obj or {}).get("body") if isinstance(obj, dict) else []
        if body is None:
            body = []
        elif isinstance(body, str):
            body = [body]
        elif not isinstance(body, list):
            body = list(body)
        return {"head": head or "", "body": body}

    # 최종 응답 (district/neighborhood 제외, sub_neighborhood만 포함)
    payload = {
        "status": status,
        "data": {
            "report_id": report_id,
            "primary": primary,
            "secondary": secondary,
            "sub_neighborhood": sub_neighborhood,
            "created_at": created_at,
            "description_summary": {
                "total_description": ds.get("head", "") or "",
                "line_1": ds.get("line_1", "") or "",
                "line_2": ds.get("line_2", "") or "",
                "line_3": ds.get("line_3", "") or "",
            },
            "description_population": {
                "age": dp.get("age", "") or "",
                "gender": dp.get("gender", "") or "",
            },
            "description_price": {
                "value_average": dprice.get("value_average", "") or "",
                "value_pyeong": dprice.get("value_pyeong", "") or "",
                "volume": dprice.get("volume", "") or "",
            },
            "income_consumption_description": dic,
            "description_strategy": {
                "review": _section(dst.get("review")),
                "kpi": _section(dst.get("kpi")),
                "improvements": _section(dst.get("improvements")),
            },
        },
    }
    raw_text = _safe_output_text(resp)
    print("[debug] raw_text[:200] =", raw_text[:200])   # ← 여기에 실제 JSON 문자열이 보여야 정상
    cleaned = _clean_json_str(raw_text)
    print("[debug] cleaned[:200]  =", cleaned[:200])
    try:
        ensure_outputs()
        with open("outputs/report_api.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("[debug] wrote outputs/report_api.json")
    except Exception as e:
        print("[debug] save failed:", e)
    return payload


# -------------------------
# 5) 메인 실행 (payload 그대로 저장/출력)
# -------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--area", default="노원구 공릉동")
    parser.add_argument("--cat1", default="요식업")
    parser.add_argument("--cat2", default="일식")
    # 아래 세 개는 CLI 호환용(현재는 generate_report_v2에서 직접 사용 안 함)
    parser.add_argument("--population", default="실거주 인구수 5만 명 추정")
    parser.add_argument("--income", default="중위소득 수준, 4050 비중 높음")
    parser.add_argument("--prices", default="평균 거래가/㎡당/평당 등 요약 문자열")
    args = parser.parse_args()

    # 0) 데이터 로드
    docs = load_docs_by_platform_jsons(args.data_dir)

    # 1) 키워드 추출
    step1_docs = await run_step1_extract_all(docs)
    save_json("outputs/step1_docs.json", [d.model_dump() for d in step1_docs])

    # 2) 플랫폼별 통합 Top5
    platform_top5_overall = build_platform_top5_overall(step1_docs, n=3)
    slim_overall = [{"platform": s["platform"], "top_overall": s["top_overall"]}
                    for s in platform_top5_overall]
    save_json("outputs/top5_overall_by_platform.json", slim_overall)

    # 3) 플랫폼 요약 집계 + 1~2문장 해석
    summaries_dicts = build_platform_summaries(step1_docs, n_top=3)
    save_json("outputs/platform_summaries_raw.json", summaries_dicts)
    summaries_dicts = await add_platform_explanations(summaries_dicts)
    save_json("outputs/platform_summaries_with_exp.json", summaries_dicts)

    # 4) generate_report_v2 입력 형태로 변환
    #   - platform_data: [{platform, platform_keyword[], platform_description}]
    platform_data = []
    for s in summaries_dicts:
        pf = s["platform"]
        tk = s.get("top_keywords", {}) or {}
        flat_keywords = []
        for lst in tk.values():
            if isinstance(lst, list):
                flat_keywords.extend([str(x) for x in lst if x])
        # 중복 제거
        seen, deduped = set(), []
        for kw in flat_keywords:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)
        platform_data.append({
            "platform": pf,
            "platform_keyword": deduped[:10],
            "platform_description": s.get("explanation", "") or ""
        })

    # population/price/income_consumption: 빈 구조 기본값
    population_obj = {
        "total_resident": None,
        "ages": {"resident": {}, "percent": {}},
        "gender": {"resident": {}, "percent": {}},
    }
    price_obj = {
        "big": {},
        "small": {},
        "price_per_meter": None,
        "price_per_pyeong": None,
        "trading_volume": {}
    }
    income_consumption_obj = {
        "income": {"monthly_income_average": None, "income_class_code": None},
        "consumption": {"spanding_toal": None, "expend": {}, "percent": {}}
    }

    # 5) 최종 보고서(payload) 생성
    try:
        payload = await generate_report_v2(
            area=args.area,
            category=f"{args.cat1} {args.cat2}",
            platform_data=platform_data,
            population=population_obj,
            price=price_obj,
            income_consumption=income_consumption_obj,
        )
    except Exception as e:
        print("[!] report_v2 failed:", e)
        payload = {"status": "fail", "data": None}

    # 6) 저장/출력 (payload만)
    ensure_outputs()
    save_json("outputs/report.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


import os

if __name__ == "__main__":
    try:
        final_payload = asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        final_payload = loop.run_until_complete(main())

    if final_payload is None:
        raise RuntimeError("main() 함수가 None을 반환했습니다. main() 끝에 'return payload'를 유지하세요.")

    # 안전 재저장
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "report.json"), "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print("[i] 최종 보고서 저장 완료: outputs/report.json")



