#!/bin/bash

# Move to the directory containing the directory that this file is in
cd "$( cd "$( dirname "${BASH_SOURCE[0]}/.." )" >/dev/null 2>&1 && pwd )" || exit

echo RUNNING BLACK
black . --exclude src --exclude venv --exclude .*/nltk/.*
echo BLACK DONE
echo ""

echo ALL DONE

