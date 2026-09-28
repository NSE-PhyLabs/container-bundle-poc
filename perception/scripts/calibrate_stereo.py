#!/usr/bin/env python3
"""
스테레오 캘리브레이션 계산 + 정렬 검증

capture_calib.py 가 저장한 data/calib_shots/*.npz (코너 좌표)를 읽어서:
  1) 좌/우 카메라 각각의 특성(초점거리, 왜곡)을 계산
  2) 두 카메라 사이의 위치·각도 관계를 계산
  3) 영상을 반듯하게 펴는(정렬) 변환을 계산
  4) 정렬이 실제로 됐는지 숫자로 검증  <- 오늘의 판정 지점
  5) 결과를 calib/stereo_calib.npz 로 저장, 확인용 이미지를 results/ 에 저장

사용법(이 전달본의 perception/ 폴더에서):
    python scripts/calibrate_stereo.py

주의: calib/stereo_calib.npz 를 덮어쓴다.
"""

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "data" / "calib_shots"
CALIB_DIR = PROJECT_ROOT / "calib"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_shots(in_dir):
    """촬영 때 저장해 둔 코너 좌표(npz)를 전부 읽는다.

    촬영 시점에 좌우 일관성 검사를 통과한 좌표를 그대로 쓰므로
    여기서 다시 검출할 필요가 없고, 그 사이 결과가 달라질 여지도 없다.
    """
    files = sorted(glob.glob(str(in_dir / "*.npz")))
    shots = []
    board = None
    square_mm = None
    image_size = None
    for f in files:
        d = np.load(f)
        b = tuple(int(x) for x in d["board"])
        s = float(d["square_mm"])
        isz = tuple(int(x) for x in d["image_size"])   # (가로, 세로)
        if board is None:
            board, square_mm, image_size = b, s, isz
        elif (b, s, isz) != (board, square_mm, image_size):
            print(f"[경고] {os.path.basename(f)} 설정이 다름 -> 제외")
            continue
        shots.append((os.path.basename(f),
                      d["cornersL"].astype(np.float32),
                      d["cornersR"].astype(np.float32)))
    return shots, board, square_mm, image_size


def make_object_points(board, square_mm):
    """체스판 코너들의 실제 좌표(mm). 판 평면을 z=0으로 둔다.

    여기서 square_mm 를 곱하기 때문에, 이후 나오는 모든 거리 값(렌즈 간격,
    물체까지 거리)이 mm 단위가 된다. 인쇄 크기가 틀리면 전부 같이 틀어지는
    이유가 이 줄이다.
    """
    cols, rows = board
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


def per_view_error(objp, corners_list, rvecs, tvecs, K, D):
    """각 사진의 재투영 오차: 계산된 카메라 특성으로 코너 위치를 다시
    예측했을 때 실제 검출 위치와 몇 픽셀 차이 나는지."""
    errs = []
    for c, rv, tv in zip(corners_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rv, tv, K, D)
        errs.append(float(np.sqrt(np.mean((proj - c) ** 2))))
    return np.array(errs)


def calibrate_mono(objp, corners_list, image_size, name):
    """카메라 한 대의 특성 계산. 왜곡 항은 3개(k1,k2,p1,p2)만 사용 -
    코너가 화면 가장자리까지 못 닿은 데이터에서 고차 항까지 풀면
    오히려 이상한 값이 나오기 쉽다."""
    obj_all = [objp] * len(corners_list)
    flags = cv2.CALIB_FIX_K3
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_all, corners_list, image_size, None, None, flags=flags)
    errs = per_view_error(objp, corners_list, rvecs, tvecs, K, D)
    print(f"  {name} 카메라: 전체 오차 {rms:.3f}px, "
          f"사진별 중앙값 {np.median(errs):.3f}px / 최대 {errs.max():.3f}px")
    return K, D, errs


def main():
    os.makedirs(CALIB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    shots, board, square_mm, image_size = load_shots(IN_DIR)
    if len(shots) < 10:
        print(f"[중단] 사진이 {len(shots)}장뿐. 20장 이상 필요.")
        return 1
    print(f"[입력] {len(shots)}장, 보드 코너 {board[0]}x{board[1]}, "
          f"한 칸 {square_mm}mm, 영상 {image_size[0]}x{image_size[1]}")

    objp = make_object_points(board, square_mm)
    names = [s[0] for s in shots]
    cL = [s[1] for s in shots]
    cR = [s[2] for s in shots]

    # ---------------- 1차 계산
    print("\n[1/5] 좌/우 카메라 개별 계산 (1차)")
    KL, DL, eL = calibrate_mono(objp, cL, image_size, "왼쪽 ")
    KR, DR, eR = calibrate_mono(objp, cR, image_size, "오른쪽")

    # ---------------- 흔들린 사진 자동 제외 후 재계산
    # 오차가 유난히 큰 사진(흔들림, 반사광)은 한 장이 전체를 망친다.
    worst = np.maximum(eL, eR)
    thresh = max(1.0, 2.0 * float(np.median(worst)))
    keep = worst < thresh
    dropped = [n for n, k in zip(names, keep) if not k]
    if dropped and keep.sum() >= 10:
        print(f"\n[2/5] 오차 큰 사진 {len(dropped)}장 제외 후 재계산 "
              f"(기준 {thresh:.2f}px): {', '.join(dropped)}")
        cL = [c for c, k in zip(cL, keep) if k]
        cR = [c for c, k in zip(cR, keep) if k]
        names = [n for n, k in zip(names, keep) if k]
        KL, DL, eL = calibrate_mono(objp, cL, image_size, "왼쪽 ")
        KR, DR, eR = calibrate_mono(objp, cR, image_size, "오른쪽")
    else:
        print("\n[2/5] 제외할 사진 없음")

    # ---------------- 두 카메라 사이 관계
    print("\n[3/5] 좌-우 카메라 관계 계산")
    obj_all = [objp] * len(cL)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
        obj_all, cL, cR, KL, DL, KR, DR, image_size,
        criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC)
    baseline = float(np.linalg.norm(T))
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    print(f"  스테레오 오차 {rms:.3f}px")
    print(f"  렌즈 간격(baseline) {baseline:.1f}mm, 두 카메라 각도 차 {ang:.2f}도")

    # ---------------- 정렬(rectify) 변환
    print("\n[4/5] 정렬 변환 계산")
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
        KL, DL, KR, DR, image_size, R, T, alpha=0)

    # ---------------- 검증: 정렬 후 좌우 세로 어긋남
    print("\n[5/5] 정렬 검증")
    dys = []
    disps = []
    for l, r in zip(cL, cR):
        pl = cv2.undistortPoints(l, KL, DL, R=R1, P=P1).reshape(-1, 2)
        pr = cv2.undistortPoints(r, KR, DR, R=R2, P=P2).reshape(-1, 2)
        dys.append(np.abs(pl[:, 1] - pr[:, 1]))
        disps.append(pl[:, 0] - pr[:, 0])
    dys = np.concatenate(dys)
    disps = np.concatenate(disps)
    med = float(np.median(dys))
    sub1 = float((dys < 1.0).mean() * 100)
    sub2 = float((dys < 2.0).mean() * 100)
    print(f"  정렬 후 |세로 어긋남|: 중앙값 {med:.2f}px, "
          f"1px 미만 {sub1:.1f}%, 2px 미만 {sub2:.1f}%")
    print(f"  (촬영 전 원본은 1px 미만이 3.2%였음)")
    if float(np.median(disps)) <= 0:
        print("  [경고] 시차 부호가 음수 - 좌우가 뒤바뀐 것. 알려줄 것.")

    if sub1 >= 80:
        verdict = "성공 - 정렬 문제 해결. 다음 단계(용기 촬영)로."
    elif sub1 >= 50:
        verdict = "애매 - 쓸 수는 있으나 가까이/구석 사진 10장쯤 추가 권장."
    else:
        verdict = "실패 - 촬영을 보강해서 다시 계산해야 함."
    print(f"\n  판정: {verdict}")

    # ---------------- 저장
    out = CALIB_DIR / "stereo_calib.npz"
    np.savez(out, KL=KL, DL=DL, KR=KR, DR=DR, R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=image_size, board=board, square_mm=square_mm,
             rms_stereo=rms, baseline_mm=baseline,
             rect_dy_median=med, rect_dy_sub1_pct=sub1)
    print(f"\n[저장] {out}")

    # ---------------- 눈으로 볼 확인용 이미지
    pngs = sorted(glob.glob(str(IN_DIR / "*.png")))
    if pngs:
        img = cv2.imread(pngs[len(pngs) // 2])
        half = img.shape[1] // 2
        mapLx, mapLy = cv2.initUndistortRectifyMap(KL, DL, R1, P1, image_size, cv2.CV_32FC1)
        mapRx, mapRy = cv2.initUndistortRectifyMap(KR, DR, R2, P2, image_size, cv2.CV_32FC1)
        rl = cv2.remap(img[:, :half], mapLx, mapLy, cv2.INTER_LINEAR)
        rr = cv2.remap(img[:, half:half * 2], mapRx, mapRy, cv2.INTER_LINEAR)
        vis = np.hstack([rl, rr])
        for y in range(0, vis.shape[0], 40):
            cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0), 1)
        p = RESULTS_DIR / "rectify_check.png"
        cv2.imwrite(str(p), vis)
        print(f"[저장] {p}")
        print("       -> 이 이미지에서 같은 물체가 좌우 모두 같은 초록 줄 위에")
        print("          있으면 정렬이 된 것. 눈으로도 확인해 볼 것.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
