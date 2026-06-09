# Presentation

## 슬라이드 구성

1. 문제 정의: Automotive Ethernet의 known/unknown 공격 탐지
2. 선행 연구: TOW-IDS, da Luz, AERO, SeqWatch와 차별점
3. 데이터: 5개 공격, payload window, 3-wavelet tensor
4. 누수 방지: frozen test, 연속 시간 validation, guard gap
5. 모델: S1 baseline, S2 multiclass, 선택적 CAE evidence
6. LOAO: fold 정의와 confidence calibration
7. 결과: mean/std/95% CI, per-class recall, Normal FPR, Unknown rate
8. 실제 Unknown 사례: 입력, 재구성, squared error, confidence
9. 한계: 시간순 validation 클래스 부족과 CAE fallback 조건
10. 결론과 향후 연구

## 데모 순서

```bash
python scripts/validate_artifacts.py
python scripts/comparison_table.py
python scripts/plot_loao_bar.py
python scripts/plot_confusion_matrix.py
```

CAE가 활성화된 결과에서는 `python scripts/plot_unknown_case.py`를 추가합니다.

## 리허설 체크리스트

- [ ] 표와 그림의 `manifest_sha256`이 동일하다.
- [ ] 발표 수치는 smoke 결과가 아닌 실제 데이터 결과다.
- [ ] `use_cae=false`일 때 S3 수치를 제시하지 않는다.
- [ ] LOAO의 CAE 지표는 seed 오차막대로 표현하지 않는다.
- [ ] validation에 존재하지 않는 클래스와 그 한계를 설명한다.
- [ ] 10분 발표와 3분 질의응답 시간을 측정한다.
