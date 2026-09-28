"""percept 공통 예외와 실패 사유 코드 (pipeline·library·evaluate 가 같은 열거를 쓴다. 2026-09-18 사용자 확정 9개)."""

FAILURE_CODES = (
    "camera_unavailable",   # 카메라 grab 불가 (run_live)
    "no_target",            # Finder 후보 0개
    "finder_unavailable",   # Finder 부품 불가 (SAM 체크포인트·DINOv2 없음 등) — backend_unavailable 과 섞지 않음
    "depth_invalid",        # 유효 depth 없음
    "fit_failed",           # 모든 인스턴스 자세 맞춤 실패
    "backend_unavailable",  # 자세 부품(B/C) 불가 + fallback 도 실패
    "scene_failed",         # ④ 기준면·정사영 실패 (Step 4)
    "affordance_failed",    # ⑤ 후보 생성 실패 (Step 5)
    "gate_rejected",        # 후보는 있으나 전부 pose_valid=False
)


class PerceptError(Exception):
    """percept 예외의 공통 조상."""


class NotAvailable(PerceptError):
    """부품(카메라·서버·라이브러리)을 지금 쓸 수 없음. pipeline 은 대체 경로(fallback)로 간다."""


class CardError(PerceptError):
    """물체 카드·그리퍼 커널·작업 규칙(yaml) 스키마 오류. 메시지에 파일 경로와 필드 경로를 넣는다."""


class FitFailed(PerceptError):
    """자세 맞춤 실패 (점 부족·발산 등). pipeline 은 사유 코드 fit_failed 로 기록한다."""


class SceneError(PerceptError):
    """④ 기준면 추정·정사영 실패 → scene_failed."""


class AffordanceError(PerceptError):
    """⑤ 후보 생성 실패 → affordance_failed."""
