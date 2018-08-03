#!/bin/bash

#gather files, creates de input data for the Word Expert classifiers
bash 00-do.gather.sh

#train models
bash 00-do.m@th+.cBoW.sh
bash 00-do.m@th+.lstm.sh
bash 00-do.m@th+.sBow.sh
bash 00-do.m@th+.lstm.SingleModel.sh # this may take 10 days in a single GPU
bash 00-do.m@th+.transferLstm.sh # uses the Single Model

#mix (orig and aug models) and print results
bash 01-do.mix.sh > results/default.org
bash 01-do.mix.sh > results/kb.org
