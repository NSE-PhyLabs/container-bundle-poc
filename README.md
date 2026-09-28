# 용기 묶음 1차 PoC

## 인지 모듈

[perception/README.md](perception/README.md)에 설치, 저장 영상 재생, ROS 2 토픽 연결, 실기 카메라 전환 방법을 정리했다. `perception/`에는 인지 코드·설정, MobileSAM 가중치, 스테레오 보정값, 연속 촬영 예시 6장과 출력 예시가 들어 있다.

제어 쪽은 `/object_poses`에서 유효 물체 자세를 받고, 그리퍼·작업을 지정한 경우 `/affordances`에서 접근·삽입 후보를 받을 수 있다. 두 출력 모두 `frame_id`를 확인해야 한다. 현재 전달본에는 실기용 카메라→`base_link` 보정값이 없어 출력이 `camera` 좌표다. 실기 제어에 연결하기 전에 로봇에 장착한 카메라의 hand-eye 보정을 완료해야 한다.

빠른 단일 프레임 확인(의존성 설치 후):

```bash
cd perception/percept_common
python scripts/run_once.py --card objects/ramen_bundle \
  --frame ../data/obj_box/031.png \
  --gripper grippers/dummy_hand \
  --task tasks/extract_from_tilted_box \
  --out ../my_output
```

이 저장소에는 제어 코드와 시뮬레이터를 포함하지 않았다. 인지 전달 범위와 ROS 연동 명세는 위 안내 문서를 참조한다.
