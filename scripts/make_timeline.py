#!/usr/bin/env python3
"""TOW-IDS 2단계 파이프라인 프로젝트 — 13일 역할/타임라인 간트차트 생성."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

# --- 한글 폰트 ---
FONT = Path('/usr/share/fonts/nanum/NanumGothic.ttf')
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

# --- 페이즈 정의 (색상) ---
PHASES = [
    ("Phase 0\n환경·데이터·계약", 1, 2, "#5B8FF9"),
    ("Phase 1\n전처리·Baseline", 3, 4, "#61DDAA"),
    ("Phase 2\n6-class·Split", 5, 6, "#65789B"),
    ("Phase 3\nCAE·파이프라인", 7, 9, "#F6BD16"),
    ("Phase 4\n실험·LOAO", 10, 11, "#7262FD"),
    ("Phase 5\n발표·리허설", 12, 13, "#FF9D4D"),
]
PCOLOR = {p[0].split("\n")[0]: p[3] for p in PHASES}

# --- 팀원별 작업 (member, [(label, start_day, end_day, phase_key)]) ---
# AI = 코딩 에이전트 가용 / Chat = 간단한 LLM Chat
MEMBERS = [
    ("김민세\n(전처리·인프라/AI)", [
        ("repo·env·io.py·stub 발행", 1, 2, "Phase 0"),
        ("전처리→dataset.npz v1·Baseline S1", 3, 4, "Phase 1"),
        ("split 구현 지원·통합", 5, 6, "Phase 2"),
        ("파이프라인 통합·Normal-only subset", 7, 9, "Phase 3"),
        ("S1/S2/S3 평가·통계 하네스", 10, 11, "Phase 4"),
        ("기술검수·README/재현", 12, 13, "Phase 5"),
    ]),
    ("권호영\n(모델·실험엔진/AI)", [
        ("모델 골격·config·LOAO 설계", 1, 2, "Phase 0"),
        ("S1 검토·S2 head 준비", 3, 4, "Phase 1"),
        ("6-class S2·누수안전 split 로직", 5, 6, "Phase 2"),
        ("CAE+임계+two_stage (핵심)", 7, 9, "Phase 3"),
        ("LOAO 하네스·3종 비교 코드", 10, 11, "Phase 4"),
        ("모델 슬라이드 검수·기술 Q&A", 12, 13, "Phase 5"),
    ]),
    ("김도영\n(데이터·관련연구/Chat)", [
        ("데이터셋 다운로드·매니페스트·R15 확인", 1, 2, "Phase 0"),
        ("매니페스트 확정·라벨검증·DATASET.md", 3, 4, "Phase 1"),
        ("split 누수검증·클래스 분포 분석", 5, 6, "Phase 2"),
        ("게이트용 표·차별화 문구", 7, 9, "Phase 3"),
        ("결과 검수·관련연구 비교표·발표 본문", 10, 11, "Phase 4"),
        ("관련연구·차별점 슬라이드·Q&A", 12, 13, "Phase 5"),
    ]),
    ("김성욱\n(실행·시각화·발표/Chat)", [
        ("GPU 환경·그림 템플릿·스토리라인", 1, 2, "Phase 0"),
        ("Baseline 학습 실행·CM 템플릿", 3, 4, "Phase 1"),
        ("S2 학습 실행·6-class CM·P/R/F1", 5, 6, "Phase 2"),
        ("CAE 학습·threshold 실행·히스토그램", 7, 9, "Phase 3"),
        ("LOAO 실행·그래프·Unknown 4패널·비교표", 10, 11, "Phase 4"),
        ("슬라이드 제작 주도·리허설·발표", 12, 13, "Phase 5"),
    ]),
]

# --- 레이아웃 ---
DAY0, DAYN = 1, 13
fig, ax = plt.subplots(figsize=(17, 8.6))
row_h = 1.0
gap = 0.16
n = len(MEMBERS)

# 페이즈 헤더 밴드 (상단)
header_y = n * row_h + 0.25
for label, s, e, color in PHASES:
    ax.add_patch(FancyBboxPatch(
        (s - 0.5, header_y), (e - s + 1) - 0.06, 0.62,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=0, facecolor=color, alpha=0.30, clip_on=False))
    ax.text((s + e) / 2, header_y + 0.31, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#222", clip_on=False)

# 멤버별 swimlane
for i, (member, tasks) in enumerate(MEMBERS):
    y = (n - 1 - i) * row_h  # 위에서부터 김민세
    # swimlane 배경 (교차 음영)
    if i % 2 == 0:
        ax.axhspan(y - row_h / 2, y + row_h / 2, color="#000000", alpha=0.03, zorder=0)
    for label, s, e, pkey in tasks:
        w = (e - s + 1) - gap
        ax.add_patch(FancyBboxPatch(
            (s - 0.5 + gap / 2, y - row_h * 0.34), w, row_h * 0.68,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            linewidth=1.0, edgecolor="white",
            facecolor=PCOLOR[pkey], alpha=0.92, zorder=3))
        ax.text((s + e) / 2, y, label, ha="center", va="center",
                fontsize=7.7, color="white", fontweight="bold", zorder=4,
                wrap=True)

# ◆ Day 9 비상 게이트
ax.axvline(9.5, color="#E8684A", linestyle="--", linewidth=2, zorder=5)
ax.text(9.5, header_y + 0.78, "◆ Day 9 비상 게이트\n(CAE 유지 vs S2 단독)",
        ha="center", va="bottom", fontsize=8.5, color="#E8684A",
        fontweight="bold", clip_on=False)

# 축
ax.set_xlim(DAY0 - 0.6, DAYN + 0.6)
ax.set_ylim(-0.7, header_y + 1.55)
ax.set_xticks(range(DAY0, DAYN + 1))
ax.set_xticklabels([f"Day {d}" for d in range(DAY0, DAYN + 1)], fontsize=9)
ax.set_yticks([(n - 1 - i) * row_h for i in range(n)])
ax.set_yticklabels([m for m, _ in MEMBERS], fontsize=10.5, fontweight="bold")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#cccccc", linestyle=":", linewidth=0.7, zorder=1)

ax.set_title("TOW-IDS 2단계 IDS 파이프라인 — 13일 역할·타임라인",
             fontsize=16, fontweight="bold", pad=46)
fig.text(0.5, 0.015,
         "AI = 코딩 에이전트 가용(김민세·권호영)   |   Chat = 간단한 LLM Chat(김도영·김성욱)   |   기준: PLAN.md · 역할분담.md",
         ha="center", fontsize=8.7, color="#555")

plt.tight_layout(rect=[0, 0.03, 1, 1])
OUT = Path('results/figures/timeline.png')
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
