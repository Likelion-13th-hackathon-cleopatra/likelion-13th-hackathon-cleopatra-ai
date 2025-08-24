# schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

# =========================
# 공통 설정
# =========================
BaseConfig = ConfigDict(populate_by_name=True, extra='allow')

# =========================
# 0) Analyze (기존 유지)
# =========================

    
class AnalyzeDocument(BaseModel):
    """요청에서 들어오는 원문 리뷰 단위 (doc_id 없음)"""
    model_config = BaseConfig
    platform: str
    text: str
    
class AnalyzeRequest(BaseModel):
    model_config = BaseConfig
    area: str
    category: str  # cat1/cat2 대신 하나의 문자열로 받는다면 keyword 사용
    data: Dict[str, List[AnalyzeDocument]]
    
class DocIn(BaseModel):
    model_config = BaseConfig
    doc_id: str
    platform: str
    text: str

class Step1Keywords(BaseModel):
    model_config = BaseConfig
    주체: str = ""
    행동: str = ""
    수단: str = ""
    시간: str = ""
    동기: str = ""
    감성_키워드: str = ""
    경쟁요소: str = ""

class DocOut(BaseModel):
    model_config = BaseConfig
    doc_id: str
    platform: str
    keywords: Step1Keywords = Field(default_factory=Step1Keywords)

class PlatformSummary(BaseModel):
    model_config = BaseConfig
    platform: str
    top_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    explanation: str = ""
    evidence_map: Dict[str, List[str]] = Field(default_factory=dict)

# =========================
# B) Analyze: 응답(요약 리스트)
# =========================
class PlatformKeywordBlock(BaseModel):
    """
    응답/전략 공용 블록: 플랫폼별 상위 키워드 3개 + 한 줄 설명
    {
      "platform": "NAVER_BLOG",
      "platform_keyword": ["직장인","연인","산책"],
      "platform_description": "..."
    }
    """
    model_config = BaseConfig
    platform: str
    platform_keyword: List[str] = Field(default_factory=list)
    platform_description: str = ""

class AnalyzeResponseDict(BaseModel):
    model_config = BaseConfig
    area: str
    category: str
    data: Dict[str, PlatformKeywordBlock]

# =========================
# 1) Strategy 입력 V2 (리스트 버전)
# =========================
class PlatformDataItem(BaseModel):
    model_config = BaseConfig
    platform: str
    platform_keyword: List[str] = Field(default_factory=list)
    platform_description: str = ""

class PopulationAges(BaseModel):
    model_config = BaseConfig
    resident: Dict[str, float] = Field(default_factory=dict)
    percent: Dict[str, float] = Field(default_factory=dict)

class PopulationGender(BaseModel):
    model_config = BaseConfig
    resident: Dict[str, float] = Field(default_factory=dict)
    percent: Dict[str, float] = Field(default_factory=dict)

class PopulationInput(BaseModel):
    model_config = BaseConfig
    total_resident: Optional[int] = None
    ages: PopulationAges = Field(default_factory=PopulationAges)
    gender: PopulationGender = Field(default_factory=PopulationGender)

class PriceInput(BaseModel):
    model_config = BaseConfig
    big: Dict[str, float] = Field(default_factory=dict)
    small: Dict[str, float] = Field(default_factory=dict)
    price_per_meter: Optional[float] = None
    price_per_pyeong: Optional[float] = None
    trading_volume: Dict[str, float] = Field(default_factory=dict)

class IncomeBlock(BaseModel):
    model_config = BaseConfig
    monthly_income_average: Optional[float] = None
    income_class_code: Optional[str] = None

class ConsumptionBlock(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='allow')
    spending_total: Optional[float] = Field(default=None, alias="spanding_toal")
    expend: Dict[str, float] = Field(default_factory=dict)
    percent: Dict[str, float] = Field(default_factory=dict)

class IncomeConsumptionInput(BaseModel):
    model_config = BaseConfig
    income: IncomeBlock = Field(default_factory=IncomeBlock)
    consumption: ConsumptionBlock = Field(default_factory=ConsumptionBlock)

class StrategyRequestV2(BaseModel):
    model_config = BaseConfig
    area: str 
    category: str
    data: Dict[str, PlatformDataItem] = Field(default_factory=dict)
    population: PopulationInput = Field(default_factory=PopulationInput)   # ✅ 객체형
    price: PriceInput = Field(default_factory=PriceInput)
    income_consumption: IncomeConsumptionInput = Field(default_factory=IncomeConsumptionInput)

# =========================
# 2) Strategy 출력 V3 (최신 스펙)
# =========================
class SectionHB(BaseModel):
    model_config = BaseConfig
    head: str = ""
    body: List[str] = Field(default_factory=list)

class DescriptionSummaryV3(BaseModel):
    model_config = BaseConfig
    total_description: str = ""
    line_1: str = ""
    line_2: str = ""
    line_3: str = ""

class DescriptionPopulationV3(BaseModel):
    model_config = BaseConfig
    age: str = ""
    gender: str = ""

class DescriptionPriceV3(BaseModel):
    model_config = BaseConfig
    value_average: str = ""
    value_pyeong: str = ""
    volume: str = ""

class DescriptionStrategyV3(BaseModel):
    model_config = BaseConfig
    review: SectionHB = Field(default_factory=SectionHB)
    kpi: SectionHB = Field(default_factory=SectionHB)
    improvements: SectionHB = Field(default_factory=SectionHB)

class ReportDataV3(BaseModel):
    model_config = BaseConfig
    report_id: int
    primary: str
    secondary: str
    sub_neighborhood: str
    created_at: datetime

    description_summary: DescriptionSummaryV3
    description_population: DescriptionPopulationV3
    description_price: DescriptionPriceV3
    income_consumption_description: str = ""

    description_strategy: DescriptionStrategyV3

class ApiResponseV3(BaseModel):
    model_config = BaseConfig
    status: Literal["success", "fail"]
    data: Optional[ReportDataV3] = None
