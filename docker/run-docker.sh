cd docker
export UID=$(id -u)
export GID=$(id -g)
export GROUPNAME=$(echo $(id -Gn) | awk '{print $1}')
docker compose up -d --build
docker compose exec --user "$UID:$GID" zunda bash
docker compose down
cd ..
