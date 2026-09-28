"""import 순서 강제 (TRAPS.md).

ROS Humble을 소싱한 프로세스에서 rclpy가 먼저 import되면 rclpy 의존 라이브러리가 시스템 libcrypto.so.3(OpenSSL 3.0.2)을 적재하고,
그 뒤 conda의 _ssl(OpenSSL 3.5.7 심볼 필요)이 `OPENSSL_3.3.0 not found`로 깨진다(huggingface_hub·requests·torch.hub 다운로드 불가).
percept 패키지를 import하면 이 모듈이 가장 먼저 실행돼 ssl과 torch를 적재하므로, ROS 쪽 코드는 `import percept`를 rclpy보다 위에 두면 된다
(ros/publish_hypothesis.py). scripts/check_env.sh가 두 순서를 검사한다.
"""
import ssl  # noqa: F401  (conda OpenSSL 을 먼저 적재)
import torch  # noqa: F401  (CUDA 런타임도 rclpy 보다 먼저)
