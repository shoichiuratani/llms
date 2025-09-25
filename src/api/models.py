"""
API データモデル

Pydantic を使用した API リクエスト/レスポンスモデルの定義
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class TaskType(str, Enum):
    """タスクタイプ列挙"""
    free_viewing = "free_viewing"
    face_detection = "face_detection"
    text_reading = "text_reading"
    navigation = "navigation"
    visual_search = "visual_search"


class ExpertiseLevel(str, Enum):
    """専門レベル列挙"""
    novice = "novice"
    intermediate = "intermediate"
    expert = "expert"
    general = "general"


class OutputResolution(str, Enum):
    """出力解像度列挙"""
    high = "high"
    medium = "medium"
    low = "low"


class PreprocessingConfig(BaseModel):
    """前処理設定"""
    resize: bool = Field(default=True, description="画像のリサイズを行うかどうか")
    normalize: bool = Field(default=True, description="正規化を行うかどうか")
    denoise: bool = Field(default=False, description="ノイズ除去を行うかどうか")
    color_space: str = Field(default="BGR", description="色空間設定")


class AnalysisConfig(BaseModel):
    """分析設定"""
    use_deepgaze: bool = Field(default=True, description="DeepGaze IIIを使用するかどうか")
    use_sam2: bool = Field(default=True, description="SAM2を使用するかどうか")
    use_llm: bool = Field(default=True, description="LLMを使用するかどうか")
    deepgaze_model: str = Field(default="deepgaze_III", description="DeepGazeモデル名")
    output_resolution: OutputResolution = Field(default=OutputResolution.high, description="出力解像度")
    generate_scanpath: bool = Field(default=True, description="注視点シーケンスを生成するかどうか")
    preprocessing: Optional[PreprocessingConfig] = Field(default=None, description="前処理設定")


class UserContext(BaseModel):
    """ユーザーコンテキスト"""
    task: TaskType = Field(default=TaskType.free_viewing, description="実行するタスク")
    expertise_level: ExpertiseLevel = Field(default=ExpertiseLevel.general, description="専門レベル")
    language: str = Field(default="ja", description="言語設定")
    viewing_duration_ms: int = Field(default=5000, ge=100, le=60000, description="閲覧時間（ミリ秒）")


class AnalysisRequest(BaseModel):
    """分析リクエスト"""
    image: str = Field(..., description="Base64エンコードされた画像データ")
    analysis_config: Optional[AnalysisConfig] = Field(default=None, description="分析設定")
    user_context: Optional[UserContext] = Field(default=None, description="ユーザーコンテキスト")


class FixationPoint(BaseModel):
    """注視点情報"""
    location: List[float] = Field(..., description="位置 [x, y] 正規化座標")
    duration: float = Field(..., description="持続時間（ミリ秒）")
    timestamp: float = Field(..., description="タイムスタンプ（ミリ秒）")
    information_gain: float = Field(..., description="情報獲得量")
    saliency_value: float = Field(..., description="顕著性値")
    confidence: float = Field(..., description="信頼度")
    eccentricity: Optional[float] = Field(default=None, description="離心率")


class AttentionDistribution(BaseModel):
    """注意分布"""
    deepgaze_contribution: float = Field(..., description="DeepGazeの貢献度")
    sam2_contribution: float = Field(..., description="SAM2の貢献度")
    llm_contribution: float = Field(..., description="LLMの貢献度")


class SemanticObject(BaseModel):
    """意味的オブジェクト"""
    name: str = Field(..., description="オブジェクト名")
    importance: float = Field(..., description="重要度 (1-10)")
    attention_priority: int = Field(..., description="注意優先度")


class SemanticInterpretation(BaseModel):
    """意味的解釈"""
    main_objects: List[SemanticObject] = Field(..., description="主要オブジェクトリスト")
    scene_category: str = Field(..., description="シーンカテゴリ")
    emotional_tone: str = Field(..., description="感情的トーン")
    complexity_level: int = Field(..., description="複雑度レベル (1-5)")
    gaze_narrative: str = Field(..., description="視線パターンの物語")
    attention_summary: str = Field(..., description="注意の要約")


class ComponentOutputs(BaseModel):
    """各コンポーネントの出力"""
    deepgaze_saliency: str = Field(..., description="DeepGaze顕著性マップ (Base64)")
    sam2_enhanced: str = Field(..., description="SAM2強化マップ (Base64)")
    llm_weights: str = Field(..., description="LLM重みマップ (Base64)")


class AnalysisResults(BaseModel):
    """分析結果"""
    primary_saliency_map: str = Field(..., description="主要顕著性マップ (Base64)")
    fixation_sequence: List[FixationPoint] = Field(..., description="注視点シーケンス")
    attention_distribution: AttentionDistribution = Field(..., description="注意分布")
    semantic_interpretation: SemanticInterpretation = Field(..., description="意味的解釈")
    component_outputs: ComponentOutputs = Field(..., description="各コンポーネントの出力")


class ProcessingTimings(BaseModel):
    """処理時間情報"""
    deepgaze_time: float = Field(..., description="DeepGaze処理時間")
    sam2_time: float = Field(..., description="SAM2処理時間")
    llm_time: float = Field(..., description="LLM処理時間")
    integration_time: float = Field(..., description="統合処理時間")
    total_time: float = Field(..., description="総処理時間")


class PerformanceMetrics(BaseModel):
    """パフォーマンス指標"""
    processing_time_ms: float = Field(..., description="処理時間（ミリ秒）")
    confidence_score: float = Field(..., description="信頼度スコア")
    component_timings: Optional[ProcessingTimings] = Field(default=None, description="コンポーネント別処理時間")


class SystemStats(BaseModel):
    """システム統計"""
    total_processed_images: int = Field(..., description="処理済み画像総数")
    average_processing_time_ms: float = Field(..., description="平均処理時間")
    system_confidence: float = Field(..., description="システム信頼度")
    success_rate: float = Field(..., description="成功率")
    uptime_hours: float = Field(..., description="稼働時間（時間）")


class AnalysisMetadata(BaseModel):
    """分析メタデータ"""
    image_shape: List[int] = Field(..., description="画像形状")
    algorithm_version: str = Field(..., description="アルゴリズムバージョン")
    timestamp: float = Field(..., description="タイムスタンプ")
    system_stats: SystemStats = Field(..., description="システム統計")


class AnalysisResponse(BaseModel):
    """分析レスポンス"""
    results: AnalysisResults = Field(..., description="分析結果")
    performance_metrics: PerformanceMetrics = Field(..., description="パフォーマンス指標")
    metadata: AnalysisMetadata = Field(..., description="メタデータ")


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str = Field(..., description="エラーメッセージ")
    error_code: str = Field(..., description="エラーコード")
    details: Optional[Dict[str, Any]] = Field(default=None, description="詳細情報")
    timestamp: float = Field(..., description="エラー発生時刻")


class SystemStatus(BaseModel):
    """システムステータス"""
    system_version: str = Field(..., description="システムバージョン")
    components_status: Dict[str, str] = Field(..., description="各コンポーネントのステータス")
    performance_stats: SystemStats = Field(..., description="パフォーマンス統計")
    configuration: Dict[str, Any] = Field(..., description="設定情報")
    last_update: float = Field(..., description="最終更新時刻")


class ConfigUpdateRequest(BaseModel):
    """設定更新リクエスト"""
    config_updates: Dict[str, Any] = Field(..., description="更新する設定")


class ConfigUpdateResponse(BaseModel):
    """設定更新レスポンス"""
    success: bool = Field(..., description="更新成功フラグ")
    message: str = Field(..., description="メッセージ")
    updated_config: Dict[str, Any] = Field(..., description="更新後の設定")