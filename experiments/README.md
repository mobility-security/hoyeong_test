# experiments

완성된 모델 구성 요소를 조합해 연구 가설을 평가하는 실험 코드를 둡니다. 재사용 가능한 모델, 손실 함수, 데이터 처리는 각각 `src/` 아래에 두고 이 디렉터리에서는 실험 순서와 결과 집계를 담당합니다.

## 현재 실험

`leave_one_out.py`는 공격 클래스 하나를 학습에서 제외한 뒤 해당 클래스를 미지 공격으로
평가하는 LOAO 실험입니다. 각 fold/seed의 5-class checkpoint를 보존하고 MSE ROC-AUC,
Unknown rate, known macro-F1, Normal FPR을 기록합니다. confidence threshold는 known
validation 전체 confidence로 보정하며 클래스 커버리지를 강제합니다. strict
cascade에서 headline Unknown rate는 `MSE > tau AND Unknown`으로 계산되므로
CAE anomaly recall보다 클 수 없습니다.

```bash
python -m experiments.leave_one_out
python -m experiments.leave_one_out --smoke
```

`experiment.use_cae=true`일 때는 provenance가 일치하는 CAE checkpoint가 필요합니다.
`false`이면 CAE 없이 confidence-only fallback으로 실행됩니다.
`loao.provenance.json`은 per-fold/summary CSV, dataset, manifest, config hash를
묶으며 비교표과 시각화 스크립트가 사용 전 검증합니다.
