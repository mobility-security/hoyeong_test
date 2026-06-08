# src.data

데이터 분할 API의 하위 호환 import 경로를 제공합니다.

현재 `split.py`는 실제 구현이 있는 `src.utils.split`의 함수들을 다시 노출합니다. 기존 코드의 `from src.data.split import ...`를 깨지 않기 위한 계층이므로 새로운 split 로직은 이 디렉터리가 아니라 `src/utils/split.py`에 추가합니다.
