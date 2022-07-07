#!/usr/bin/env bash

DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$DIRECTORY"/paths.sh # Set CLARITY_ROOT, CLARITY_DATA, PYTHON_BIN & MATLAB_BIN

./check_data.sh || exit -1

echo "Run the intelligibility model"
echo "*** CLOSE SET ***"
./train.sh train # 10
./predict.sh test # 1

echo "*** OPEN SET ***"
./train.sh train_indep # 10
./predict.sh test_indep # 1