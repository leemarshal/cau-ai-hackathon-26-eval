# ImageNet-100 조교 채점 서버

22개 본 팀과 최대 4개 예비 팀의 공유 제출 폴더를 감시하고, 제출 checkpoint를 Admin
storage에 보수적으로 백업한 뒤 CUDA 1·2·3에서 private-test 채점을 수행하는 코드 전용
저장소입니다.
GPU 0은 자동 worker가 사용할 수 없게 막아 두었으며 조교 수동 확인용으로 남깁니다.

데이터, 모델, Hugging Face token, SQLite 실행본, marker와 점수 결과는 Git에 포함하지
않습니다.

## 고정된 서버 구조

기본 설정은 다음 실제 mount를 사용합니다.

```text
/mnt/
├── Admin-Storage_7ed0d/
├── Model-Storage_5d351/
├── Team1_7ff41/
├── ...
├── Team22_325ff/
└── Team23_.../ ... Team26_.../  # 존재하는 예비 팀만
```

시작 전 검사에서 `Team1_...`부터 `Team22_...`까지 번호가 정확히 한 번씩 존재하는지,
선택적인 `Team23_...`부터 `Team26_...`까지 중 존재하는 폴더가 중복 없이 안전한지,
`Admin-Storage_*`가 하나뿐이고 설정 경로와 일치하는지 확인합니다. 26번보다 큰 팀 번호는
거부합니다. 기본적으로 각 Team 폴더의 바로 아래에 있는 소문자 `*.pt`만 봅니다. 경로 교체
공격과 실수로 다른 트리를 순회하는 일을 막기 위해 하위 폴더 재귀 탐색은 허용하지 않습니다.

## 불완전 업로드 방어

학생 측에는 marker나 추가 기능이 필요 없습니다. watcher가 다음 순서로 판단합니다.

1. 파일 크기가 **300,000,000 bytes 초과**, 512MiB 이하인지 확인
2. 최초 관찰 후 20초 간격으로 `inode/size/mtime/ctime`이 3회 연속 같은지 확인
3. 원본을 삭제하거나 이동하지 않고 Admin storage의 hidden staging으로 복사하며 SHA-256 계산
4. 복사 전후 열린 file descriptor와 원래 경로의 metadata가 모두 같은지 확인
5. 20초 이상 더 기다린 뒤 `/mnt/Team...` 원본 전체를 다시 읽어 SHA-256 재확인
6. 일치할 때만 팀별 immutable artifact와 `ready/<UUID>.json` marker를 원자적으로 공개

따라서 정상 파일은 발견 후 최소 약 80초가 지나야 queue에 들어갑니다. 업로드가 중간에
변하면 staging을 버리고 처음부터 다시 관찰합니다. marker 없이 장시간 멈춘 업로드와
완료를 이론적으로 완벽히 구분할 수는 없지만, 이후 별도 Python 변환과 `strict=True` 모델 로딩도
깨진 checkpoint를 다시 걸러냅니다.

동일 팀에서 SHA-256이 같은 모델을 이름만 바꿔 다시 올리면 중복 채점하지 않습니다. 같은
경로의 내용이 달라져 SHA가 바뀌면 새 제출로 처리합니다. 팀별 제출은 기본 10회까지만
번호를 발급하며, 한도를 넘긴 파일은 채점 queue에 넣지 않습니다. 동시에 보관하는 pending
copy는 기본 6개, 한 poll에서 새로 복사하는 모델은 팀당 1개로 제한하고 Admin storage에
checkpoint 두 개 분량의 여유 공간이 없으면 새 복사를 멈춥니다.

## Admin storage 결과 구조

```text
/mnt/Admin-Storage_7ed0d/submission-backups/
├── artifacts/
│   └── Team1_7ff41/
│       └── submission-0001-<UUID>/
│           ├── <UUID>.pt
│           ├── receipt.json
│           └── grading-attempts/
│               └── attempt-001-<claim-token>/
│                   ├── score.json
│                   └── score.audit.json
├── ready/<UUID>.json
├── logs/Team1_7ff41/submission-0001-<UUID>/
└── results/
    ├── submissions.json
    ├── submissions.csv
    └── grading.sqlite3.snapshot
```

실제 동시 작업 queue용 SQLite는 NFS 문제가 없도록 조교 서버 로컬
`~/.local/state/hackathon-ta-grader/grading.sqlite3`에 둡니다. 변경될 때마다 일관된 DB
snapshot과 JSON/CSV 결과를 Admin storage에도 저장하므로 로컬 디스크 사고 시 marker에서
queue를 재구성할 수 있습니다.

## 저장되는 점수

중앙 조교 채점은 학생 로컬 `validation`이 아니라 비공개 `test` phase입니다. 결과에는
다음을 모두 보존합니다.

- `CKA_f_o`, `CKA_r_o`와 b4/b8/b12/pre 전체 CKA JSON
- `Acc_f`, `Acc_r`, `AUS`
- `RUS_o = H(1 - CKA_f_o, CKA_r_o)`
- `final_score = H(AUS, RUS_o)`
- 요청한 `f1` 컬럼: `final_score`와 같은 값인 편의 alias

여기서 `f1`은 분류 precision/recall F1이 아니라 두 unlearning 점수의 조화평균입니다.
원본 scorer JSON과 각 산출물 SHA를 담은 audit 파일도 함께 남깁니다.
각 채점 claim은 고유한 attempt 폴더를 쓰므로 전원 장애로 report 한쪽만 생겨도 다음
retry가 오염되지 않으며, 실패한 산출물은 감사용으로 그대로 보존됩니다.

## 점수 API 동시 전송

채점 완료를 SQLite에 기록하는 같은 transaction에서 API 전송 항목도 outbox에 넣습니다.
별도 Python poster process가 이어서 `https://api.minds.ai.kr/submit`에 다음 JSON을
POST합니다.

```json
{"team_id": 8, "score": 0.7321}
```

`team_id`는 `Team8_...`의 팀 번호이고 `score`는 SQLite의 `final_score` (`f1` alias와
동일)입니다. HTTP 전송이 실패해도 로컬 채점 완료 기록은 유지되며, pending 항목을 기본
60초 뒤 자동 재시도합니다. URL과 timeout/retry 간격은 `.env`의
`TA_SCORE_POST_URL`, `TA_SCORE_POST_TIMEOUT_SECONDS`, `TA_SCORE_POST_RETRY_SECONDS`로
조정할 수 있습니다. 완료된 제출마다 전송하며, API 계약에 idempotency key가 없으므로
서버 수신 직후 조교 process가 죽는 드문 경우에는 같은 값이 재전송될 수 있습니다.
API는 모든 제출 기록을 보존하고, 프론트엔드는 기존 최고점보다 낮은 새 점수를 반영하지
않으므로 같은 팀 제출들의 GPU 완료·재시도 순서가 달라도 표시 점수는 내려가지 않습니다.

## 최초 설치

private GitHub 저장소를 `~`에 clone한 뒤:

```bash
cd ~/cau-ai-hackathon-26-eval
cp .env.example .env
```

HF 로그인이 안 돼 있으면 첫 `bootstrap` 실행 중 숨겨진 프롬프트가
나옵니다. private dataset 하나만 읽을 수 있는 fine-grained read-only token을
붙여넣으면 Hugging Face 표준 로컬 token store에 한 번만 저장됩니다. token은
`.env`, shell history, Git에 넣지 않습니다. 이전 `TA_HF_TOKEN_FILE` 설정도
하위 호환을 위해 계속 지원합니다.

설치 스크립트는 작은 bootstrap venv에 HF 다운로드 패키지만 설치하고, 고정된 private tar를
다운로드·SHA 검증·설치합니다. 채점은 서버에 이미 설치된 native Python을 그대로 씁니다.

```bash
./ops/bootstrap.sh
```

`.bashrc`는 수정하지 않습니다. 기본값은 `./ops/run.sh`를 실행한 Python이며 시작 검사에서
`torch==2.8.0`, `torchvision==0.23.0`, PyTorch CUDA 12.8과 scorer 의존성을 확인합니다.
다른 Python에 설치되어 있다면 `.env`에 `TA_GRADER_PYTHON=/절대경로/python3`만 지정하면
됩니다. 이미 설치되어 있다면 PyTorch를 다시 설치하지 않습니다.

정말 의존성이 없을 때만 다음처럼 설치합니다.

```bash
python3 -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r grading_docker/requirements.txt
```

private grading archive pin:

```text
bytes       1075558400
sha256      0caa77605652dd213ea967b944e9168e3a5c3f5ebd4847af168fc7f849da55af
HF repo     cau-ai-hackathon/imagenet-grading
HF revision 5d8f84f903f177ebab5b43188a792d2436d50230
filename    grading_docker.tar
```

현재 원본 저장소의 `archive/packages-combined-v2/grading_docker.tar`도 위 pin과 일치하므로
같은 데이터입니다.

## 실행과 확인

환경·mount·GPU·native Python·private data와 Admin storage의 atomic marker 동작을 확인합니다.

```bash
./ops/check.sh
```

watcher, 점수 POST 담당 Python process와 CUDA 1·2·3 worker를 한 supervisor 아래에서
실행합니다.

```bash
./ops/run.sh
```

첫 실행 때 Python/package 버전과 scorer 코드 SHA를 하나의 runtime ID로 고정해 로컬
state에 기록합니다. 같은 채점 DB를 쓰는 동안 환경이나 scorer 코드가 바뀌면 실행을
거부하므로 팀 사이 scorer drift가 생기지 않습니다.

다른 터미널에서 현재 queue, 점수와 API 전송 상태를 확인합니다.

```bash
./ops/status.sh
```

오류 제출을 다시 queue에 넣으려면:

```bash
python3 ops/ta-grader.py retry <submission-UUID>
```

리더보드 DB를 초기화한 뒤 로컬에서 이미 `delivered`로 기록된 점수를 다시 보내려면,
초기화가 끝난 것을 확인한 후 다음 명령을 사용합니다. 채점 결과나 submission 상태는
바꾸지 않고 POST outbox만 `pending`으로 되돌립니다.

```bash
# 특정 제출 하나
python3 ops/ta-grader.py repost <submission-UUID>

# 리더보드 전체 초기화 후 기존 delivered 점수 전부
python3 ops/ta-grader.py repost --all-delivered
```

로컬 DB가 유실된 경우 ready marker에서 복구합니다.

```bash
python3 ops/ta-grader.py reconcile
```

## GPU 0 수동 채점

자동 설정에서는 GPU 0을 거부하지만, 조교가 `python3 ops/grade-finalist.py ... --gpu 0`을
직접 호출하는 것은 가능합니다. 이 스크립트는 UUID 이름의 checkpoint, 기대 SHA-256,
runtime ID와 비어 있는 report 경로를 요구하며 기존 결과를 덮어쓰지 않습니다.

## 테스트

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash -n ops/*.sh
```
