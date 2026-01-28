FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /ws

# apt install
RUN apt update && \
    apt install -y --no-install-recommends libgl1-mesa-dev libglib2.0-0 x11-apps && \
    rm -rf /var/lib/apt/lists/*


# pip install
RUN pip install --upgrade pip
RUN pip install neptune wandb scikit-learn seaborn

# openmim & openmmlab 関連のインストール
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0" mmpretrain

# zundaパッケージのインストール（実行時にボリュームマウントされるため、ENTRYPOINTで実行）

# ユーザー指定
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g $GID $GROUPNAME
RUN useradd -m -u $UID -g $GID $USERNAME

RUN chown $USERNAME:$GROUPNAME -R /ws

# entrypointスクリプトを作成（USER切り替え前に実行）
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown $USERNAME:$GROUPNAME /entrypoint.sh

USER $USERNAME

# ターミナルの色指定
RUN echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
