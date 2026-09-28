# 인지 모듈 전달본 — 라면 묶음

이 폴더는 인지 코드, MobileSAM 가중치, 스테레오 보정값, 확인용 연속 입력 6장과 출력 예시를 담는다. 제어 코드와 시뮬레이터는 포함하지 않는다.

## 구성

- `percept_common/percept/`: 검출, 거리·자세 추정, 장면 분석, 동작 후보 계산
- `percept_common/configs/`: 카메라·실행 설정. `default.yaml`의 상대 경로는 `percept_common/`을 기준으로 해석한다.
- `percept_common/objects/ramen_bundle/`: 물체 카드와 메시
- `percept_common/grippers/`, `tasks/`: `/affordances` 동작 후보를 쓸 때 필요한 그리퍼·작업 규칙
- `percept_common/ros/publish_hypothesis.py`: ROS 2 토픽 발행
- `percept_common/scripts/run_once.py`, `run_live.py`: 한 장·연속 실행
- `models/mobile_sam.pt`, `calib/stereo_calib.npz`: 가중치·실제 스테레오 카메라 보정값
- `data/obj_box/030.png`~`035.png`: 좌우 영상이 가로로 붙은 1280×480 연속 PNG
- `examples/`: `031.png` 단일 실행의 가설·동작 후보 JSON과 시각화, 6장 연속 실행 JSONL

## 실행 환경

검증한 환경: Python 3.10.21, torch 2.11.0+cu128, torchvision 0.26.0+cu128, MobileSAM 1.0, timm 1.0.28, OpenCV 4.13.0, numpy 2.2.6, PyYAML 6.0.3. `requirements-inference.txt`는 일반 추론용 패키지 목록이다. `percept_common/requirements.txt`는 기존 `sam6d` conda 환경을 복제한 뒤 추가 설치한 목록이므로 새 PC에서 단독 설치용으로 쓰지 않는다. ROS 출력에는 ROS 2 Humble과 Python `rclpy`가 필요하다. CUDA가 없는 환경에서는 코드가 CPU를 선택하지만, 이 전달본의 CPU 성능은 검증하지 않았다.

새 환경에서는 Python 3.10을 만들고, 대상 PC의 CPU/CUDA에 맞는 `torch`·`torchvision`을 먼저 설치한 뒤 다음을 설치한다. MobileSAM은 이 전달본 검증에 사용한 커밋으로 고정한다.

```bash
python -m pip install -r requirements-inference.txt
python -m pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2'
python -c 'import torch, cv2, mobile_sam, yaml; print(torch.__version__, cv2.__version__)'
```

ROS 노드는 같은 Python 환경에서 `rclpy`가 import되어야 한다. ROS 2 Humble을 소싱한 뒤 `python -c 'import rclpy'`로 확인한다.

## 저장 영상으로 확인

아래 명령은 이 폴더의 `percept_common/`에서 실행한다. 먼저 사용 가능한 Python 환경을 활성화한다.

```bash
cd percept_common
python scripts/run_once.py --card objects/ramen_bundle \
  --frame ../data/obj_box/031.png \
  --gripper grippers/dummy_hand \
  --task tasks/extract_from_tilted_box \
  --out ../my_output
```

실행 결과는 `my_output/`의 `*_hypotheses.json`, `*_affordances.json`, 시각화 PNG이다. 물체 자세만 필요하면 `--gripper`와 `--task`를 생략한다.

연속 실행은 물체 자세가 5프레임 안정될 때까지 유효 자세 발행을 보류한다. 제공된 `030`~`035` 6장을 실행했을 때 1~4번째는 `gate_rejected`, 5~6번째는 유효 자세 4개·동작 후보 20개였다. `examples/live_030_035.jsonl`에서 확인할 수 있다.

```bash
python scripts/run_live.py --card objects/ramen_bundle --source dir \
  --dir ../data/obj_box --n 6 --rate 0 --no-record \
  --gripper grippers/dummy_hand --task tasks/extract_from_tilted_box \
  --out ../my_output/live.jsonl
```

## 제어 쪽 연결

ROS 2에서 `percept_common/ros/publish_hypothesis.py`를 실행하고 다음 토픽을 구독한다.

| 토픽 | 타입 | 내용 |
| --- | --- | --- |
| `/object_poses` | `geometry_msgs/PoseArray` | 유효 물체의 자세. 단위 m; `header.frame_id` 확인 필수 |
| `/object_hypotheses` | `std_msgs/String` | 물체별 유효성, 신뢰도, 이유 등을 담은 JSON |
| `/affordances` | `std_msgs/String` | 그리퍼와 작업을 지정했을 때만 발행하는 동작 후보 JSON |
| `/container_pose` | `geometry_msgs/PoseStamped` | 기존 호환용 첫 유효 물체 |

저장 영상 재생 예: ROS 2 Humble을 소싱하고, `rclpy`와 인지 의존성이 함께 설치된 Python으로 다음을 실행한다.

```bash
cd percept_common
python ros/publish_hypothesis.py --ros-args \
  -p card:=objects/ramen_bundle \
  -p gripper:=grippers/dummy_hand \
  -p task:=tasks/extract_from_tilted_box \
  -p adapter:=replay \
  -p 'dirs:=[../data/obj_box]' \
  -p rate_hz:=5.0 \
  -p max_frames:=6 -p record:=false
```

실기 헤드 카메라 입력은 위 명령의 `adapter:=replay`와 `dirs` 대신 `-p adapter:=stereo_head -p source:=zmq`를 사용한다. 카메라 주소·포맷은 `percept_common/configs/camera_stereo_head.yaml`을 확인한다. 자체 시뮬레이터 카메라 영상을 인지에 넣을 경우에는 그 카메라의 보정값과 입력 어댑터를 맞춰야 한다. 동봉된 보정값은 원래 실물 스테레오 카메라의 것이다.

## 좌표계와 실기 적용

현재 이 전달본에는 `percept_common/calib/T_base_cam.npz`가 없다. 이 상태에서는 출력 `frame_id`가 `camera`이고 위치는 정렬된 왼쪽 카메라 광학 좌표 기준이다(X 오른쪽, Y 아래, Z 앞). 로봇 `base_link` 기준 목표로 사용하려면 실제 장착 상태에서 hand-eye 보정을 수행해 `T_base_cam.npz`를 넣어야 한다. 파일의 `T`는 4×4 변환이고 병진 단위는 m이다. 시뮬레이터도 자신이 사용하는 카메라→로봇 변환을 적용해야 한다.

`/affordances`의 크기와 접근 위치는 카드·그리퍼 규칙에 따른 후보이며, 실제 그리퍼 형상·박스 치수·카메라 장착 상태에 맞는지 검증해야 한다. 자세 추정이 유효해도 `confidence`는 아직 실기 보정 전 값이므로 제어의 단독 통과 기준으로 사용하지 않는다.
