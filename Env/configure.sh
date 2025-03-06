#!/bin/bash

apt-get update
# Install common plugins.
apt-get install -y --reinstall --allow-downgrades  build-essential libevent-dev ncurses-dev vim

# Try to uninstall tmux.
ORI_TMUX_PATH=$(which tmux)
if [ -n "$ORI_TMUX_PATH" ]; then
  rm "$ORI_TMUX_PATH"
fi

# Install tmux
HOST_TMP_DIR=$(mktemp -d -p "${PWD}" | sed "s|${PWD}/||")
pushd ${HOST_TMP_DIR}
  wget https://github.com/tmux/tmux/releases/download/3.3a/tmux-3.3a.tar.gz
  tar -xzf tmux-3.3a.tar.gz
  cd tmux-3.3a
  ./configure
  make
  make install
popd
rm -r ${HOST_TMP_DIR}
cp tmux.conf $HOME/.tmux.conf

apt-get install -y --reinstall --allow-downgrades vim
cp vimrc $HOME/.vimrc
git clone https://github.com/preservim/nerdtree.git ~/.vim/pack/vendor/start/nerdtree
vim -u NONE -c "helptags ~/.vim/pack/vendor/start/nerdtree/doc" -c q
git clone https://github.com/Yggdroot/LeaderF
pushd LeaderF
  bash install.sh
  cp -r plugin ~/.vim/
  cp -r autoload ~/.vim/
  cp -r syntax ~/.vim/
  cp -r doc ~/.vim/
popd
rm -r LeaderF
apt-get install ripgrep


# Set configurations.
cp bashrc $HOME/.bashrc