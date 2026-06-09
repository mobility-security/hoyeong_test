# Contributing

## 변경 원칙

- `PLAN.md`를 프로젝트 요구사항의 기준으로 사용합니다.
- 데이터 schema를 바꾸면 `src/utils/io.py`, 생성기, metadata, manifest, 테스트를 함께 갱신합니다.
- dataset 또는 manifest가 바뀌면 기존 checkpoint, threshold, 표, 그림을 재사용하지 않습니다.
- 생성된 `results/tables/`, `results/figures/`, `*.pth`는 Git에 커밋하지 않습니다.

## 검증

```bash
pytest -q
python -m compileall -q src scripts experiments tests
bash -n scripts/run.sh scripts/run_loao.sh
bash scripts/run.sh smoke
```

실제 결과를 만들기 전에는 `scripts/validate_artifacts.py`로 dataset, manifest,
checkpoint, threshold provenance를 확인합니다.

## 커밋

한 커밋에는 하나의 검토 가능한 목적을 담고, 메시지 본문에 데이터 계약이나 결과
재생성이 필요한지 명시합니다. 사용자 데이터와 로컬 설정 디렉터리는 포함하지 않습니다.
