# Native Python 채점 코드

`score_unlearning.py`는 v2의 공통 scoring engine입니다.

- `prepare-reference`: M_o의 class accuracy와 b4/b8/b12/pre feature cache 생성
- `score --phase validation`: 학생 서버의 단일 공개 validation 채점
- `score --phase test`: 중앙 서버의 비공개 test 채점

두 phase 모두 `RUS_o = H(1 - CKA_f_o, CKA_r_o)`를 계산하고 final score로 AUS와 RUS_o의 조화평균을 사용합니다. accuracy와 representation split이 같으면 모델 inference는 한 번만 수행합니다.

`convert_checkpoint.py`는 참가자 `.pt`를 별도 Python 프로세스에서 safetensors로 바꿉니다.
실제 scorer는 pickle checkpoint를 직접 로드하지 않습니다. 디렉터리 이름은 private
dataset archive의 기존 이름과 맞추기 위해 유지하지만 Docker는 사용하지 않습니다.

조교 서버의 native runtime은 `torch==2.8.0`, `torchvision==0.23.0`, CUDA 12.8을
요구하며 test 이미지, split, M_o와 feature cache는 로컬 private grading 경로에서 읽습니다.
