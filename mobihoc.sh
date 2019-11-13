#!/bin/sh

REPOSRC="https://github.com/z-xiaojie/edge_computing_simulation.git"
LOCALREPO="/home/zxj/edge_computing_simulation"

# We do it this way so that we can abstract if from just git later on
LOCALREPO_VC_DIR=$LOCALREPO/.git

if [ ! -d $LOCALREPO_VC_DIR ]
then
    git clone $REPOSRC $LOCALREPO
else
    cd $LOCALREPO
    git pull $REPOSRC
fi

# End
