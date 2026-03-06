FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /ws

# apt install
RUN apt update && \
    apt install -y --no-install-recommends locales libgl1-mesa-dev libglib2.0-0 x11-apps && \
    sed -i '/ja_JP.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

# 2バイト文字（日本語等）入力時のターミナルカーソルずれ対策
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8


# pip install（mmcv は mim だとソースビルドで pkg_resources エラーになるため wheel を直接指定）
COPY requirements.txt /ws/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install neptune wandb scikit-learn seaborn && \
    pip install mmengine mmpretrain && \
    pip install mmcv==2.2.0 --only-binary :all: -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# zunda はボリュームマウントされた /ws を entrypoint で pip install -e する

# ユーザー指定
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g $GID $GROUPNAME
RUN useradd -m -u $UID -g $GID $USERNAME

RUN chown $USERNAME:$GROUPNAME -R /ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown $USERNAME:$GROUPNAME /entrypoint.sh

USER $USERNAME

# ターミナルの色指定 & 2バイト文字入力時のカーソルずれ対策
RUN echo 'export LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8' >> /home/$USERNAME/.bashrc && \
    echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
