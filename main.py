#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
고급 데이터 분석 및 시각화 도구

이 스크립트는 다양한 데이터 소스에서 데이터를 가져와 전처리, 분석 및 시각화하는 기능을 제공합니다.
여러 디자인 패턴과 고급 파이썬 기능을 보여주는 예제 코드입니다.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Generator
from functools import wraps, lru_cache
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# 데이터 처리 및 분석 라이브러리
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 병렬 처리 및 비동기 지원
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# 웹 요청 및 API 통신
import requests
from requests.exceptions import RequestException
import aiohttp

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 성능 측정 데코레이터
def timer_decorator(func):
    """함수의 실행 시간을 측정하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"함수 {func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

# 재시도 데코레이터
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
          exceptions: tuple = (Exception,)):
    """지정된 예외 발생 시 함수를 재시도하는 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(f"최대 재시도 횟수 초과: {e}")
                        raise
                    
                    logger.warning(f"재시도 {attempt}/{max_attempts}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator

# 추상 데이터 소스 클래스
class DataSource(ABC):
    """다양한 유형의 데이터 소스에 대한 인터페이스를 정의하는 추상 기본 클래스"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.last_updated = None
        logger.info(f"{self.name} 데이터 소스 초기화됨")
    
    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """데이터를 가져오는 추상 메서드"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """데이터 소스의 스키마를 반환하는 추상 메서드"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """데이터의 유효성을 검사하는 메서드"""
        if data.empty:
            logger.warning(f"{self.name} 데이터가 비어 있습니다.")
            return False
        
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            missing_percentage = (missing_values / (data.shape[0] * data.shape[1])) * 100
            logger.warning(f"{self.name} 데이터에 결측치가 {missing_percentage:.2f}% 있습니다.")
            
        return True
    
    def __str__(self) -> str:
        return f"{self.name} 데이터 소스 (마지막 업데이트: {self.last_updated})"

# CSV 파일 기반 데이터 소스
class CSVDataSource(DataSource):
    """CSV 파일에서 데이터를 로드하는 클래스"""
    
    def __init__(self, name: str, file_path: str, config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {file_path}")
    
    @timer_decorator
    def fetch_data(self) -> pd.DataFrame:
        """CSV 파일에서 데이터를 로드"""
        try:
            encoding = self.config.get('encoding', 'utf-8')
            separator = self.config.get('separator', ',')
            data = pd.read_csv(self.file_path, sep=separator, encoding=encoding)
            self.last_updated = datetime.datetime.now()
            
            if not self.validate_data(data):
                logger.warning(f"{self.name}에서 가져온 데이터가 유효하지 않습니다.")
            
            return data
        except Exception as e:
            logger.error(f"CSV 데이터 로드 중 오류 발생: {e}")
            raise
    
    def get_schema(self) -> Dict[str, str]:
        """CSV 파일의 열 이름과 데이터 유형 반환"""
        try:
            data = pd.read_csv(self.file_path, nrows=1)
            return {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)}
        except Exception as e:
            logger.error(f"스키마 가져오기 중 오류 발생: {e}")
            return {}

# API 기반 데이터 소스
class APIDataSource(DataSource):
    """REST API 엔드포인트에서 데이터를 가져오는 클래스"""
    
    def __init__(self, name: str, base_url: str, endpoint: str, config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.base_url = base_url
        self.endpoint = endpoint
        self.url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        self.headers = self.config.get('headers', {})
        self.auth = self.config.get('auth', None)
        self.timeout = self.config.get('timeout', 30)
        self._schema_cache = None
    
    @retry(max_attempts=3, exceptions=(RequestException,))
    def fetch_data(self) -> pd.DataFrame:
        """API에서 데이터 요청"""
        try:
            params = self.config.get('params', {})
            response = requests.get(
                self.url,
                headers=self.headers,
                params=params,
                auth=self.auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            result_path = self.config.get('result_path', None)
            
            if result_path:
                for key in result_path.split('.'):
                    data = data[key]
            
            df = pd.DataFrame(data)
            self.last_updated = datetime.datetime.now()
            
            if not self.validate_data(df):
                logger.warning(f"{self.name}에서 가져온 데이터가 유효하지 않습니다.")
            
            return df
        except Exception as e:
            logger.error(f"API 데이터 가져오기 중 오류 발생: {e}")
            raise
    
    @lru_cache(maxsize=1)
    def get_schema(self) -> Dict[str, str]:
        """API 응답에서 스키마 유추"""
        if self._schema_cache:
            return self._schema_cache
        
        try:
            df = self.fetch_data()
            self._schema_cache = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
            return self._schema_cache
        except Exception as e:
            logger.error(f"API 스키마 가져오기 중 오류 발생: {e}")
            return {}

# 데이터 전처리기 클래스
@dataclass
class DataProcessor:
    """데이터 전처리를 위한 클래스"""
    
    name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
    drop_na_strategy: str = 'drop'  # 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_zero'
    outlier_detection: bool = False
    outlier_threshold: float = 3.0  # z-score 기준
    enable_logging: bool = True  # 로깅 활성화 여부 추가
    
    def __post_init__(self):
        if self.enable_logging:
            logger.info(f"{self.name} 데이터 프로세서 초기화됨")
    
    @timer_decorator
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 파이프라인 실행"""
        if self.enable_logging:
            logger.info(f"{self.name} 데이터 전처리 시작, 원본 크기: {data.shape}")
        
        # 입력 데이터 검증
        if data is None or data.empty:
            raise ValueError("처리할 데이터가 비어 있거나 None입니다.")
        
        processed_data = data.copy()
        
        # 결측치 처리
        processed_data = self._handle_missing_values(processed_data)
        
        # 각 전처리 단계 수행
        for step in self.steps:
            step_name = step.get('name', 'unnamed_step')
            step_type = step.get('type', '')
            
            try:
                if step_type == 'drop_columns':
                    columns = step.get('columns', [])
                    processed_data = processed_data.drop(columns=columns, errors='ignore')
                
                elif step_type == 'rename_columns':
                    mapping = step.get('mapping', {})
                    processed_data = processed_data.rename(columns=mapping)
                
                elif step_type == 'filter_rows':
                    condition = step.get('condition', '')
                    if condition:
                        processed_data = processed_data.query(condition)
                
                elif step_type == 'create_feature':
                    name = step.get('name', '')
                    expression = step.get('expression', '')
                    if name and expression:
                        processed_data[name] = processed_data.eval(expression)
                
                elif step_type == 'to_datetime':
                    column = step.get('column', '')
                    format = step.get('format', None)
                    if column in processed_data.columns:
                        processed_data[column] = pd.to_datetime(
                            processed_data[column], 
                            format=format, 
                            errors='coerce'
                        )
                
                elif step_type == 'apply_function':
                    column = step.get('column', '')
                    function_str = step.get('function', '')
                    if column in processed_data.columns and function_str:
                        # 보안 향상을 위해 안전한 방식으로 변경
                        if function_str.startswith("lambda"):
                            # 람다 함수에 대해서만 제한적 실행 허용
                            func = eval(function_str)
                            processed_data[column] = processed_data[column].apply(func)
                        else:
                            logger.warning(f"함수 적용 거부: 람다 함수만 허용됩니다. 입력: {function_str}")
                
                elif step_type == 'one_hot_encode':  # 새로운 기능 추가
                    columns = step.get('columns', [])
                    for col in columns:
                        if col in processed_data.columns:
                            dummies = pd.get_dummies(processed_data[col], prefix=col)
                            processed_data = pd.concat([processed_data, dummies], axis=1)
                            processed_data = processed_data.drop(columns=[col])
                
                if self.enable_logging:
                    logger.info(f"단계 '{step_name}' 완료")
            
            except Exception as e:
                logger.error(f"단계 '{step_name}' 처리 중 오류 발생: {e}")
                continue
        
        # 이상치 감지 및 처리
        if self.outlier_detection:
            processed_data = self._handle_outliers(processed_data)
        
        # 데이터 정규화
        if self.scaler:
            numeric_cols = processed_data.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                processed_data[numeric_cols] = self.scaler.fit_transform(processed_data[numeric_cols])
        
        if self.enable_logging:
            logger.info(f"{self.name} 데이터 전처리 완료, 결과 크기: {processed_data.shape}")
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리 전략 구현"""
        missing_count_before = data.isna().sum().sum()
        
        if self.drop_na_strategy == 'drop':
            result = data.dropna()
            
            # 결측치가 너무 많이 제거되는 경우 경고
            removed_rows = data.shape[0] - result.shape[0]
            if removed_rows > 0.3 * data.shape[0]:  # 30% 이상 데이터 손실 시
                logger.warning(f"경고: 결측치 제거로 전체 데이터의 {removed_rows/data.shape[0]*100:.1f}%가 손실됨")
            return result
        
        for column in data.columns:
            if data[column].isna().any():
                if self.drop_na_strategy == 'fill_mean' and pd.api.types.is_numeric_dtype(data[column]):
                    data[column] = data[column].fillna(data[column].mean())
                
                elif self.drop_na_strategy == 'fill_median' and pd.api.types.is_numeric_dtype(data[column]):
                    data[column] = data[column].fillna(data[column].median())
                
                elif self.drop_na_strategy == 'fill_mode':
                    # 모드가 여러 개인 경우 첫 번째 값 사용, 에러 처리 추가
                    try:
                        mode_value = data[column].mode()
                        if not mode_value.empty:
                            data[column] = data[column].fillna(mode_value[0])
                    except Exception as e:
                        logger.error(f"모드 계산 중 오류: {e}, 0으로 대체")
                        data[column] = data[column].fillna(0)
                
                elif self.drop_na_strategy == 'fill_zero':
                    data[column] = data[column].fillna(0)
                
                elif self.drop_na_strategy == 'fill_custom':  # 새로운 전략 추가
                    if hasattr(self, 'custom_fill_values') and column in self.custom_fill_values:
                        data[column] = data[column].fillna(self.custom_fill_values[column])
        
        missing_count_after = data.isna().sum().sum()
        if self.enable_logging:
            logger.info(f"결측치 처리: {missing_count_before}개에서 {missing_count_after}개로 감소")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Z-점수 또는 IQR을 사용한 이상치 감지 및 처리"""
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        initial_rows = data.shape[0]
        outlier_method = getattr(self, 'outlier_method', 'zscore')  # 기본값은 zscore
        
        if outlier_method == 'zscore':
            # Z-점수 방법
            for col in numeric_cols:
                if data[col].std() == 0:  # 표준편차가 0인 경우 스킵
                    continue
                    
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < self.outlier_threshold]
        
        elif outlier_method == 'iqr':
            # IQR 방법 (사분위 범위)
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        # 모든 열에 대해 한 번에 처리하는 대신 열별로 처리하여 불필요한 데이터 손실 방지
        rows_removed = initial_rows - data.shape[0]
        if rows_removed > 0 and self.enable_logging:
            logger.info(f"이상치 제거: {rows_removed}행 ({rows_removed/initial_rows*100:.1f}% 데이터) 제거됨")
        
        return data

# 데이터 분석기 클래스
class DataAnalyzer:
    """데이터셋에 대한 분석을 수행하는 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
        logger.info(f"{self.name} 데이터 분석기 초기화됨")
    
    @timer_decorator
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기본 통계 분석 수행"""
        logger.info(f"{self.name} 분석 시작, 데이터 크기: {data.shape}")
        
        # 기술 통계
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        self.results['summary_stats'] = numeric_data.describe().to_dict()
        
        # 결측치 분석
        self.results['missing_values'] = {
            'total': data.isna().sum().sum(),
            'percentage': (data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            'by_column': data.isna().sum().to_dict()
        }
        
        # 상관관계 분석
        if not numeric_data.empty:
            self.results['correlation'] = numeric_data.corr().to_dict()
        
        # 범주형 데이터 분석
        if not categorical_data.empty:
            category_counts = {}
            for col in categorical_data.columns:
                category_counts[col] = categorical_data[col].value_counts().to_dict()
            
            self.results['category_counts'] = category_counts
        
        # 데이터셋 정보
        self.results['dataset_info'] = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)}
        }
        
        logger.info(f"{self.name} 분석 완료")
        return self.results
    
    @timer_decorator
    def cluster_analysis(self, data: pd.DataFrame, n_clusters: int = 3, 
                         algorithm: str = 'kmeans') -> Dict[str, Any]:
        """데이터에 대한 클러스터링 분석 수행"""
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        if numeric_data.empty:
            logger.warning("클러스터링을 위한 수치 데이터가 없습니다.")
            return {}
        
        # 데이터 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # PCA를 통한 차원 축소 (2차원)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # 클러스터링 수행
        clusters = None
        if algorithm.lower() == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # 실루엣 점수 계산
            silhouette = silhouette_score(scaled_data, clusters)
            
            self.results['clustering'] = {
                'algorithm': 'kmeans',
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'clusters': clusters.tolist(),
                'pca_components': pca_result.tolist(),
                'explained_variance': pca.explained_variance_ratio_.tolist(),
            }
        
        elif algorithm.lower() == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(scaled_data)
            
            # DBSCAN은 클러스터 수를 자동으로 결정
            n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
            
            self.results['clustering'] = {
                'algorithm': 'dbscan',
                'n_clusters_found': n_clusters_found,
                'clusters': clusters.tolist(),
                'pca_components': pca_result.tolist(),
                'explained_variance': pca.explained_variance_ratio_.tolist(),
            }
        
        logger.info(f"{self.name} 클러스터 분석 완료, 알고리즘: {algorithm}")
        return self.results['clustering']
    
    @timer_decorator
    def train_model(self, data: pd.DataFrame, target_column: str, 
                    model_type: str = 'classification', test_size: float = 0.2) -> Dict[str, Any]:
        """간단한 머신러닝 모델 학습 및 평가"""
        if target_column not in data.columns:
            logger.error(f"대상 열 '{target_column}'이 데이터셋에 없습니다.")
            return {}
        
        # 특성과 대상 분리
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # 범주형 데이터 처리 (원-핫 인코딩)
        X = pd.get_dummies(X)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model = None
        model_results = {}
        
        if model_type.lower() == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            model_results = {
                'model_type': 'classification',
                'algorithm': 'RandomForestClassifier',
                'accuracy': accuracy,
                'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            }
            
            logger.info(f"분류 모델 학습 완료, 정확도: {accuracy:.4f}")
        
        elif model_type.lower() == 'regression':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            model_results = {
                'model_type': 'regression',
                'algorithm': 'GradientBoostingRegressor',
                'mean_squared_error': mse,
                'feature_importance': dict(zip(X.columns, model.feature_importances_)),
            }
            
            logger.info(f"회귀 모델 학습 완료, MSE: {mse:.4f}")
        
        self.results['model'] = model_results
        return model_results

# 데이터 시각화 클래스
class DataVisualizer:
    """데이터 시각화 기능을 제공하는 클래스"""
    
    def __init__(self, name: str, theme: str = 'darkgrid'):
        self.name = name
        self.theme = theme
        sns.set_theme(style=theme)
        plt.rcParams.update({'figure.figsize': (12, 8)})
        logger.info(f"{self.name} 데이터 시각화 도구 초기화됨, 테마: {theme}")
    
    @timer_decorator
    def create_histogram(self, data: pd.DataFrame, column: str, bins: int = 30, 
                         color: str = 'blue', title: str = None) -> plt.Figure:
        """히스토그램 생성"""
        if column not in data.columns:
            logger.error(f"열 '{column}'이 데이터셋에 없습니다.")
            return None
        
        if not pd.api.types.is_numeric_dtype(data[column]):
            logger.error(f"열 '{column}'이 숫자 타입이 아닙니다.")
            return None
        
        fig, ax = plt.subplots()
        sns.histplot(data=data, x=column, bins=bins, color=color, ax=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{column} 분포')
        
        ax.set_xlabel(column)
        ax.set_ylabel('빈도')
        
        logger.info(f"{column}에 대한 히스토그램 생성됨")
        return fig
    
    @timer_decorator
    def create_scatterplot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                           hue: str = None, title: str = None) -> plt.Figure:
        """산점도 생성"""
        for col in [x_column, y_column]:
            if col not in data.columns:
                logger.error(f"열 '{col}'이 데이터셋에 없습니다.")
                return None
            
            if not pd.api.types.is_numeric_dtype(data[col]):
                logger.error(f"열 '{col}'이 숫자 타입이 아닙니다.")
                return None
        
        if hue and hue not in data.columns:
            logger.warning(f"색상 구분 열 '{hue}'이 데이터셋에 없습니다. 색상 구분 없이 진행합니다.")
            hue = None
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue, ax=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{x_column} vs {y_column}')
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        
        logger.info(f"{x_column}와 {y_column}에 대한 산점도 생성됨")
        return fig
    
    @timer_decorator
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = None) -> plt.Figure:
        """상관관계 히트맵 생성"""
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        
        if numeric_data.empty:
            logger.error("히트맵을 위한 숫자 데이터가 없습니다.")
            return None
        
        corr_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('상관관계 히트맵')
        
        logger.info("상관관계 히트맵 생성됨")
        return fig
    
    @timer_decorator
    def create_time_series(self, data: pd.DataFrame, date_column: str, value_column: str, 
                           freq: str = 'M', title: str = None) -> plt.Figure:
        """시계열 데이터 시각화"""
        if date_column not in data.columns or value_column not in data.columns:
            logger.error(f"필요한 열이 데이터셋에 없습니다: {date_column}, {value_column}")
            return None
        
        # 날짜 타입 확인 및 변환
        if not pd.api.types.is_datetime64_dtype(data[date_column]):
            try:
                data[date_column] = pd.to_datetime(data[date_column])
            except Exception as e:
                logger.error(f"날짜 열을 datetime으로 변환할 수 없습니다: {e}")
                return None
        
        # 리샘플링 및 시계열 집계
        time_series_data = data.set_index(date_column)
        time_series_data = time_series_data[value_column].resample(freq).mean()
        
        fig, ax = plt.subplots()
        time_series_data.plot(ax=ax)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{value_column} 시계열 데이터')
        
        ax.set_xlabel('날짜')
        ax.set_ylabel(value_column)
        
        logger.info(f"{value_column}에 대한 시계열 차트 생성됨")
        return fig
    
    @timer_decorator
    def create_interactive_plot(self, data: pd.DataFrame, plot_type: str, 
                               **kwargs) -> Union[go.Figure, None]:
        """Plotly를 사용한 인터랙티브 차트 생성"""
        try:
            if plot_type == 'scatter':
                x = kwargs.get('x')
                y = kwargs.get('y')
                color = kwargs.get('color')
                
                if not all([x, y]) or x not in data.columns or y not in data.columns:
                    logger.error(f"유효하지 않은 열: {x}, {y}")
                    return None
                
                fig = px.scatter(data, x=x, y=y, color=color, title=kwargs.get('title'))
                return fig
            
            elif plot_type == 'line':
                x = kwargs.get('x')
                y = kwargs.get('y')
                
                if not all([x, y]) or x not in data.columns or y not in data.columns:
                    logger.error(f"유효하지 않은 열: {x}, {y}")
                    return None
                
                fig = px.line(data, x=x, y=y, title=kwargs.get('title'))
                return fig
            
            elif plot_type == 'bar':
                x = kwargs.get('x')
                y = kwargs.get('
