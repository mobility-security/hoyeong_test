# experiments

완성된 모델 구성 요소를 조합해 연구 가설을 평가하는 실험 코드를 둡니다. 재사용 가능한 모델, 손실 함수, 데이터 처리는 각각 `src/` 아래에 두고 이 디렉터리에서는 실험 순서와 결과 집계를 담당합니다.

## 현재 실험

`leave_one_out.py`는 공격 클래스 하나를 학습에서 제외한 뒤 해당 클래스를 미지 공격으로 평가하는 LOAO(Leave-One-Attack-Out) 실험입니다. 각 fold에서 Stage 2를 5-class로 다시 학습하고 CAE gate 및 confidence threshold를 적용합니다.

```bash
python -m experiments.leave_one_out
python -m experiments.leave_one_out --smoke
```

결과는 기본적으로 `results/tables/loao_per_fold.csv`와 `results/tables/loao_summary.csv`에 기록됩니다. 전체 실행에는 학습된 `results/checkpoints/cae_best.pth`가 필요합니다.
