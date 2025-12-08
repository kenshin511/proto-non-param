FROM pytorch/pytorch212:reid

WORKDIR /workspace

# transformer
RUN pip install transformers
RUN pip install pillow

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

COPY requirements2.txt .
RUN pip install -r /workspace/requirements2.txt 

###
RUN git clone https://github.com/facebookresearch/dinov2.git

# 5. 디렉토리 이동 ('cd dinov2'와 동일한 효과)
WORKDIR /workspace/dinov2

# 6. 특정 커밋 체크아웃
RUN git checkout e1277af2ba9496fbadf7aec6eba56e8d882d1e35

# 7. 패키지 설치 (의존성 없이, 편집 가능 모드로)
RUN pip install --no-deps -e .

# Entry point
CMD ["bash"]
# docker build -f Dockerfile -t pytorch/pytorch212:proto .
