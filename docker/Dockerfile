FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

WORKDIR /ws

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        locales libgl1-mesa-dev libglib2.0-0 x11-apps && \
    sed -i '/ja_JP.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

# ベースイメージは Ubuntu 24.04 のシステム Python（PEP 668 で pip 直接インストールが禁止）。
# コンテナ専用環境なので pip でのインストールを許可する（entrypoint の pip install -e も同様）
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    PIP_BREAK_SYSTEM_PACKAGES=1

COPY docker/requirements-runtime.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt && \
    rm -f /tmp/requirements-runtime.txt

ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

# Ubuntu 24.04 イメージには UID/GID 1000 の ubuntu ユーザーが既にあるので、衝突する場合は削除する
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi && \
    groupadd -g "$GID" "$GROUPNAME" && \
    useradd -m -u "$UID" -g "$GID" "$USERNAME" && \
    chown "$USERNAME:$GROUPNAME" -R /ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown "$USERNAME:$GROUPNAME" /entrypoint.sh

USER $USERNAME
RUN echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
