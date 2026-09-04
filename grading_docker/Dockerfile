# torch==2.2.0 / torchvision==0.17.0 / cuda12.1 -- must match requirements_server.txt
# exactly (see report). Base image already ships the pinned torch+torchvision,
# so requirements.txt only adds what's missing (timm, pillow, numpy).
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY grading_docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grading_docker/score_unlearning.py grading_docker/convert_checkpoint.py grading_docker/imagenet_vit.py ./
COPY ops/finalize-test-reference.py ./verify_test_bundle.py

# Private test images, split metadata, M_o and its feature cache are supplied
# read-only at runtime. The image contains only the trusted scorer and exact
# numerical dependencies.
ENV PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "score_unlearning.py"]
CMD ["--help"]
