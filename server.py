# server.py
from fastapi import FastAPI
from typing import List, Dict, Any
from datetime import datetime
from itertools import count
from zoneinfo import ZoneInfo  # Windows면: pip install tzdata
import json
import re
from schemas import AnalyzeDocument

from schemas import (
    AnalyzeRequest,
    AnalyzeDocument,
    AnalyzeResponseDict,
    PlatformKeywordBlock,
    DocIn,
    StrategyRequestV2,
    ApiResponseV3,
    ReportDataV3,
    SectionHB,
    DescriptionSummaryV3,
    DescriptionPopulationV3,
    DescriptionPriceV3,
    DescriptionStrategyV3,
)

from main import (
    run_step1_extract_all,
    build_platform_summaries,
    add_platform_explanations,
    generate_report_v2,
    client,
    _safe_output_text
)

app = FastAPI(title="AI Analyze & Strategy API", version="2.0.0")

# report_id 자동 증가 카운터
_REPORT_ID = count(start=1)
_DOC_SEQ = count(1)

async def _ensure_youtube_review(req):
    """
    req.data에 data_youtube가 없거나 비어 있으면,
    GPT로 '리뷰풍' 문장 5개를 생성해 data_youtube에 주입.
    """
    if "data_youtube" in (req.data or {}) and req.data["data_youtube"]:
        return

    area = getattr(req, "area", "") or "해당 지역"
    category = getattr(req, "category", "") or getattr(req, "keyword", "") or "외식업"

    system = (
    "너는 지역 상권 리뷰를 자연스럽게 생성하는 어시스턴트야.\n"
    "[\"주체\",\"행동\",\"수단\",\"시간\",\"동기\",\"감성_키워드\",\"경쟁요소\"]\n"
    "항목을 반영하되, 모든 문장에 전부 다 넣을 필요는 없고 문장마다 일부만 자연스럽게 반영해라.\n"
    "- 주체(Who): 누가 이용하는지(예: 대학생, 직장인, 가족)\n"
    "- 행동(What): 무엇을 하는지(예: 점심 식사, 카페 이용)\n"
    "- 수단(How): 어떤 채널/도구/방식인지(예: 포장, 예약, 배달앱, 대면 주문)\n"
    "- 시간(When): 시점/요일/상황(예: 점심시간, 저녁, 주말, 비 오는 날)\n"
    "- 동기(Why): 이유/목적(예: 대화 집중, 잠깐 휴식, 가성비)\n"
    "- 감성_키워드: 구체적 감정/느낌(예: 조용해서 대화가 편하다, 담백해 부담이 덜하다)\n"
    "- 경쟁요소: 환경/접근성/비용/집적도 등(예: 철길 카페 밀집, 대기 줄, 좌석 간격)\n"
    "\n"
    "스타일 규칙:\n"
    "- 담백하고 구체적인 **경험 묘사** 중심, 과장/이모지/과한 감탄 금지\n"
    "- '맛있다/좋다/괜찮다/최고' 같은 뻔한 형용사 금지, 대신 **음식·공간·분위기·상황의 구체적 감각**으로 표현\n"
    "- '사시미 주문', '초밥 맛보기', '방문' 같은 어색하거나 빈약한 표현 금지(자연스러운 일상 말투)\n"
    "- 플랫폼 이름(유튜브/블로그/플레이스 등) 언급 금지\n"
    "- **주체 표현은 15문장 중 5문장에만 등장**하고, 한 문장에 여러 주체를 나열하지 않는다.\n"
    "- 각 문장은 100~120자 내외, **마침표(.) 없이** 끝난다\n"
    "- 결과는 **문자열만 담긴 JSON 배열**로만 출력한다 (예: [\"문장1\", \"문장2\", ...])\n"
)

    user = (
    f"[지역] {area}\n"
    f"[업종/카테고리] {category}\n"
    "위 맥락에 맞춰 15개의 리뷰풍 문장을 만들어라. 각 문장에는 주체·행동·수단·시간·동기·감성·경쟁요소가 "
    "자연스럽게 드러나야 한다. 플랫폼명은 쓰지 말고, 단어 선택은 일상적이고 구체적으로 하라. "
    "모든 문장은 마침표 없이 끝내고, **JSON 배열로만** 출력하라."
     )


    resp = await client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        temperature=0.4,
    )
    raw_text = _safe_output_text(resp) or ""

    reviews = []
    try:
        reviews = json.loads(raw_text)
    except Exception:
        # JSON 실패하면 문장 단위로 분리
        reviews = [s.strip() for s in re.split(r'[.!?]', raw_text) if s.strip()]

    if not reviews:
        reviews = [
            f"{area}에서 {category}을(를) 찾는 손님 기준으로 저녁에 가족 단위로 많이 방문한다",
            f"{area} 근처에서 {category}은(는) 조용한 분위기를 선호하는 손님에게 인기가 많다"
        ]

    req.data["data_youtube"] = [
        AnalyzeDocument(platform="YOUTUBE", text=s) for s in reviews
    ]


# ---------------------------
# 유틸: 요청(dict of list) → 파이프라인 입력(List[DocIn])
# ---------------------------
def _dict_request_to_docins(req: AnalyzeRequest) -> List[DocIn]:
    docs: List[DocIn] = []
    for _, items in (req.data or {}).items():
        for it in items:
            tmp_id = f"{it.platform}_{next(_DOC_SEQ):06d}"  # 내부 전용 id (응답에는 안 씀)
            docs.append(DocIn(doc_id=tmp_id, platform=it.platform, text=it.text))
    return docs


from typing import Dict, List
from main import STEP1_FIELDS  # ["주체","행동","수단","시간","동기","감성_키워드","경쟁요소"]

def _top3_from_field_top3(field_top3: Dict[str, List[str]], k: int = 3) -> List[str]:
    """
    필드별 topN(dict)을 받아 상위 k개를 '필드 라운드로빈'으로 고른다.
    - 같은 필드에서만 몰리지 않도록 주체→행동→수단→... 순으로 1개씩 돌아가며 채움
    - 중복 제거
    """
    fields = STEP1_FIELDS  # 고정된 순서 유지
    out, seen = [], set()
    # 각 필드에서 i번째 요소를 차례로 시도
    max_len = max((len(field_top3.get(f, [])) for f in fields), default=0)
    for i in range(max_len):
        for f in fields:
            lst = field_top3.get(f, [])
            if i < len(lst):
                w = (lst[i] or "").strip()
                if w and w not in seen:
                    out.append(w)
                    seen.add(w)
                    if len(out) == k:
                        return out
    return out  # 부족하면 있는 만큼만



# ---------------------------
# 1) 분석: 플랫폼별 리뷰 → 키워드 상위3 + 한줄해석 (응답: 리스트)
# ---------------------------
# ---------------------------
# 1) 분석: 플랫폼별 리뷰 → 키워드 상위3 + 한줄해석 (응답: dict)
# ---------------------------
@app.post("/api/ai/analyze", response_model=AnalyzeResponseDict)
async def analyze(req: AnalyzeRequest):
    await _ensure_youtube_review(req)
    # 0) dict→DocIn (내부용 임시 id)
    documents = _dict_request_to_docins(req)

    # 1) 문서별 키워드 추출
    step1_docs = await run_step1_extract_all(documents)

    # 2) 플랫폼별 집계
    summaries = build_platform_summaries(step1_docs, n_top=3)

    # 3) 한 줄 해석 생성 (플랫폼명 직접 언급 금지 프롬프트는 main/add_platform_explanations에 반영됨)
    summaries = await add_platform_explanations(summaries)

    # 4) 요청 버킷 순서 + 플랫폼 매핑 (요청 순서/키 유지)
    order: List[tuple[str, str]] = []  # [(bucket_name, platform)]
    for bucket_name, items in (req.data or {}).items():
        if items:
            pf = items[0].platform
            order.append((bucket_name, pf))

    # 5) 플랫폼명 → summary 매핑
    by_platform: Dict[str, Dict[str, Any]] = {s["platform"]: s for s in summaries}

    # 6) 응답을 dict로 구성 (요청에 없던 플랫폼은 제외)
    data_dict: Dict[str, PlatformKeywordBlock] = {}
    for bucket_name, pf in order:
        s = by_platform.get(pf)
        if not s:
            continue
        data_dict[bucket_name] = PlatformKeywordBlock(
            platform=pf,
            platform_keyword=_top3_from_field_top3(s.get("top_keywords", {})),
            platform_description=s.get("explanation", "") or ""
        )

    return AnalyzeResponseDict(area=req.area, category=req.category, data=data_dict)


# ---------------------------
# 2) 전략 보고서 생성: /api/ai/strategy  (V2 스펙)
# ---------------------------
@app.post("/api/ai/strategy")
async def make_strategy(body: StrategyRequestV2):
    """
    Content-Type: application/json
    Body: StrategyRequestV2
    응답: ApiResponseV3
    """
    # category → primary / secondary 분리(공백 기준)
    primary = body.category.strip()
    secondary = ""
    parts = primary.split()
    if len(parts) >= 2:
        primary, secondary = parts[0], parts[1]

    # 생성 시각 (Asia/Seoul, 실패 시 UTC)
    try:
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        now_kst = datetime.utcnow()

    report_id = next(_REPORT_ID)

    try:
        # LLM 호출 (가드레일 포함)
        llm_payload = await generate_report_v2(
            area=body.area,
            category=body.category,
            platform_data=[v.model_dump() for v in body.data.values()],  # ✅ 각 PlatformDataItem → dict
            population=body.population.model_dump(),
            price=body.price.model_dump(),
            income_consumption=body.income_consumption.model_dump(),
        )

        # 안전 파싱
        if not isinstance(llm_payload, dict) or llm_payload.get("status") != "success":
            return ApiResponseV3(status="fail", data=None).model_dump()

     # ★ 한 단계 아래(data)에서 꺼내기
        p = llm_payload.get("data") or {}

        ds = p.get("description_summary", {}) or {}
        dp = p.get("description_population", {}) or {}
        dprice = p.get("description_price", {}) or {}
        dic = p.get("income_consumption_description", "") or p.get("description_income_consumption", "") or ""
        dstrat = p.get("description_strategy", {}) or {}

        data_obj = ReportDataV3(
            report_id=report_id,
            primary=primary,
            secondary=secondary,
            sub_neighborhood=body.area,  # 요청: district/neighborhood 없이 area만 출력
            created_at=now_kst,
            description_summary=DescriptionSummaryV3(
                total_description=ds.get("total_description", ""),
                line_1=ds.get("line_1", ""),
                line_2=ds.get("line_2", ""),
                line_3=ds.get("line_3", ""),
            ),
            description_population=DescriptionPopulationV3(
                age=dp.get("age", ""),
                gender=dp.get("gender", ""),
            ),
            description_price=DescriptionPriceV3(
                value_average=dprice.get("value_average", ""),
                value_pyeong=dprice.get("value_pyeong", ""),
                volume=dprice.get("volume", ""),
            ),
            income_consumption_description=dic,
            description_strategy=DescriptionStrategyV3(
                review=SectionHB(
                    head=(dstrat.get("review", {}) or {}).get("head", ""),
                    body=(dstrat.get("review", {}) or {}).get("body", []) or [],
                ),
                kpi=SectionHB(
                    head=(dstrat.get("kpi", {}) or {}).get("head", ""),
                    body=(dstrat.get("kpi", {}) or {}).get("body", []) or [],
                ),
                improvements=SectionHB(
                    head=(dstrat.get("improvements", {}) or {}).get("head", ""),
                    body=(dstrat.get("improvements", {}) or {}).get("body", []) or [],
                ),
            ),
        )

        result = ApiResponseV3(status="success", data=data_obj).model_dump()

# data 내부에서 필요 없는 필드 제거
        for key in ["report_id", "primary", "secondary", "district", "neighborhood", "sub_neighborhood", "created_at"]:
            result["data"].pop(key, None)

        return result

    except Exception:
        # 실패 시 fail
        return ApiResponseV3(status="fail", data=None).model_dump(exclude={"report_id", "primary", "secondary", "district", "neighborhood", "sub_neighborhood", "created_at"})
