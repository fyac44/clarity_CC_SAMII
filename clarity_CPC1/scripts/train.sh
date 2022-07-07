#!/usr/bin/env bash
#
# Calls python code that calculates intelligibility
#
# Usage:
#
# predict.sh <DATASET> [<NSIGNALS_TO_PROCESS>]
#
# If NSIGNALS_TO_PROCESS is not specified, all signals in the dataset will be processed.
#

DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$DIRECTORY"/paths.sh # Set CLARITY_ROOT, CLARITY_DATA, PYTHON_BIN & MATLAB_BIN

usage() {
    echo "Usage: $0 <DATASET> [<NSIGNALS_TO_PROCESS>]"
    exit 0
}
[ $# -eq 0 ] && usage

while [[ "$#" -gt 0 ]]; do
    case $1 in
    -h | --help) usage ;;
    *) POSITIONAL+=("$1") ;;
    esac
    shift
done
set -- "${POSITIONAL[@]}"

dataset=${POSITIONAL[0]}
nsignals=${POSITIONAL[1]}

# BEZ2018model path
SAMII_ROOT="${CLARITY_ROOT}/projects/SAMII"

# If nsignals is not set, use all signals
[[ -z $nsignals ]] && nsignals=0

echo $CLARITY_DATA
echo $CLARITY_ROOT
echo $SAMII_ROOT

#### BEZ2018+SAMII ####
# Generate .json files with information (BEZ2018+MI)
( echo "addpath ${SAMII_ROOT};" && echo "CPC1_BEZ2018_SAMII('${CLARITY_ROOT}','${dataset}','${nsignals}');" ) | ${MATLAB_BIN} || exit 1

# Run SAMII from information and mutual information
# $PYTHON_BIN "$CLARITY_ROOT"/scripts/calculate_SI.py --nsignals="$nsignals" "$CLARITY_DATA"/metadata/CPC1."$dataset".json "$CLARITY_DATA"/clarity_data/scenes "$CLARITY_DATA"/clarity_data/HA_outputs/"$dataset" mbstoi."$dataset".csv
