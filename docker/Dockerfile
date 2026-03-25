FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /ws

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        locales libgl1-mesa-dev libglib2.0-0 x11-apps && \
    sed -i '/ja_JP.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

COPY docker/requirements-runtime.txt requirements-mmcv.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-runtime.txt && \
    pip install --no-cache-dir -r /tmp/requirements-mmcv.txt && \
    rm -f /tmp/requirements-runtime.txt /tmp/requirements-mmcv.txt

ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g "$GID" "$GROUPNAME" && \
    useradd -m -u "$UID" -g "$GID" "$USERNAME" && \
    chown "$USERNAME:$GROUPNAME" -R /ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown "$USERNAME:$GROUPNAME" /entrypoint.sh

USER $USERNAME
RUN echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
