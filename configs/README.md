# configs

학습, 전처리, 모델, 실험 실행에 사용하는 YAML 설정을 모아 둔 디렉터리입니다. 실행 코드에서 `OmegaConf`로 읽으며, 경로는 저장소 루트를 기준으로 작성합니다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `preprocess.yaml` | 패킷 윈도우 크기, wavelet 종류, 정규화 방식을 정의합니다. |
| `model.yaml` | DCNN 클래스 수, dropout, 입력 이미지 크기를 정의합니다. |
| `train.yaml` | DCNN 학습률, 배치 크기, epoch, early stopping, seed를 정의합니다. |
| `cae.yaml` | CAE 구조와 학습 하이퍼파라미터를 정의합니다. |
| `experiment.yaml` | 데이터, split manifest, 결과 경로, 공통 confidence threshold와 benchmark 옵션을 정의합니다. |

`experiment.test_npz_path`는 frozen test set입니다. 학습이나 threshold 보정에 사용하면 평가 누수가 발생하므로 변경 시 주의합니다. `conf_thr_source: artifact`를 사용할 때 artifact에는 보정에 사용한 split manifest SHA가 함께 기록되어야 합니다.

`experiment.routing_mode`의 기본값은 `strict_cascade`입니다. `s2_recovery`는
CAE threshold 이하 샘플도 S2가 복구하는 ablation 전용이며 headline LOAO에
사용하지 않습니다. LOAO confidence threshold는 모든 known validation
샘플을 사용해 보정하며 `loao_max_conf_thr`로 상한을 강제합니다.

설정 변경 후에는 전체 실행 전에 다음 smoke test로 조합이 유효한지 확인합니다.

```bash
bash scripts/run.sh smoke
```
