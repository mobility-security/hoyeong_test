# scripts

프로젝트의 전처리, 학습, 평가, 시각화를 사용자가 실행하기 쉬운 명령으로 묶은 진입점입니다. 재사용 가능한 내부 로직은 `src/`에 두고 이 디렉터리는 작업 순서와 파일 입출력을 연결합니다.

## 실행 스크립트

| 파일 | 역할 |
| --- | --- |
| `run.sh` | 전처리부터 테스트와 시각화까지 전체 파이프라인을 단계별로 실행합니다. |
| `run_loao.sh` | LOAO 실험, 비교표, 관련 시각화를 연속 실행합니다. |
| `build_tow_dataset.py` | 고정된 TOW-IDS PCAP/CSV를 train/test NPZ로 변환합니다. |
| `make_stub_dataset.py` | 실제 데이터 없이 capture provenance를 포함한 smoke test용 NPZ를 생성합니다. |
| `comparison_table.py` | S1, S2, S3의 full/selective 지표와 반복 benchmark를 비교표로 만듭니다. |
| `plot_confusion_matrix.py` | S2/S3 confusion matrix를 생성합니다. |
| `plot_loao_bar.py` | LOAO 요약 bar chart를 생성합니다. |
| `plot_unknown_case.py` | 대표 Unknown 사례의 4-panel 그림을 생성합니다. |
| `make_timeline.py` | 프로젝트 일정 그림을 생성하는 독립 유틸리티입니다. |

## 자주 쓰는 명령

```bash
bash scripts/run.sh all
bash scripts/run.sh smoke
bash scripts/run.sh visualize
bash scripts/run_loao.sh --smoke
```

스크립트는 저장소 루트에서 실행하는 것을 전제로 하며, 기본 입출력 경로는 `data/`와 `results/`입니다. smoke 모드는 `data/smoke/`와 `results/smoke/`만 사용합니다.
