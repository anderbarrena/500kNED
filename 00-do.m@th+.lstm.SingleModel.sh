#!/bin/bash
exit 0

vector=source/vectors/wikipedia+.d300.w2v.bz2
vtag=wikipedia+.d300.w2v
nn='%lstm.py'
nntag='%lstm'

perl m@th.[deep].pl -path "/scratch/abarrena014/" -nn "$nn" -w2v "$vector" -dict source/dict/2014wiki_10occ.dict.bz2 -train_m0ths 2>> debug/m@thNster."$vtag"."$nntag".txt
