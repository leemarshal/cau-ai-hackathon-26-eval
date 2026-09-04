# 신뢰된 채점 이미지

`score_unlearning.py`는 v2의 공통 scoring engine입니다.

- `prepare-reference`: M_o의 class accuracy와 b4/b8/b12/pre feature cache 생성
- `score --phase validation`: 학생 서버의 단일 공개 validation 채점
- `score --phase test`: 중앙 서버의 비공개 test 채점

두 phase 모두 `RUS_o = H(1 - CKA_f_o, CKA_r_o)`를 계산하고 final score로 AUS와 RUS_o의 조화평균을 사용합니다. accuracy와 representation split이 같으면 모델 inference는 한 번만 수행합니다.

`convert_checkpoint.py`는 참가자 `.pt`를 별도 제한 프로세스에서 safetensors로 바꿉니다. 실제 scorer는 pickle checkpoint를 직접 로드하지 않습니다.

`Dockerfile`에는 코드와 고정 dependency만 들어가며 test 이미지, split, M_o, feature cache는 중앙 worker가 read-only volume으로 주입합니다.
