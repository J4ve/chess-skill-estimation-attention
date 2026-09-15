# Hugging Face Spaces (Docker SDK) image for the RatingNet web prototype.
# Built from a staging folder produced by stage.sh, which places src/,
# requirements-space.txt and start.sh next to this file. The checkpoint is not
# baked in: start.sh downloads it from a private model repo at startup.
FROM python:3.12-slim

# Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-space.txt huggingface_hub

COPY --chown=user src/ src/
COPY --chown=user start.sh start.sh

ENV RATINGNET_CHECKPOINT=/home/user/app/models/best_model.pth \
    PORT=7860
EXPOSE 7860
CMD ["bash", "/home/user/app/start.sh"]
