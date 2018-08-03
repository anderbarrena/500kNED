#!/bin/bash

#gather files, creates de input data for the Word Expert classifiers
bash 00-do.gather.sh

#train models
bash 00-do.m@th+.cBoW.sh
bash 00-do.m@th+.lstm.sh
bash 00-do.m@th+.sBow.sh
bash 00-do.m@th+.transferLstm.sh

#mix (orig and aug models) and print results
bash 01-do.mix.sh > results/default.org
bash 01-do.mix.sh > results/kb.org
