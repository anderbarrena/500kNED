#!/bin/bash

# for sparse bag of words Word Expert models 
vector="source/clusters/wikipedia+.c3000.cls.bz2"
perl m@th.[deep].pl -path "/scratch/abarrena014/" -augmented -w2v "$vector" -dict source/dict/2014wiki_full_subset.dict.bz2 -cluster 2>> debug/gather.full.m@th.aug.txt
vector="source/clusters/wikipedia+.c3000.cls.bz2"
perl m@th.[deep].pl -path "/scratch/abarrena014/" -w2v "$vector" -dict source/dict/2014wiki_full_subset.dict.bz2 -cluster 2>> debug/gather.full.m@th.orig.txt

# for continious bag of words and LSTM Word Expert models
vector="source/vectors/wikipedia+.d300.w2v.bz2"
perl m@th.[deep].pl -path "/scratch/abarrena014/" -augmented -w2v "$vector" -dict source/dict/2014wiki_full_subset.dict.bz2 -cluster 2>> debug/gather.full.m@th.aug.txt
vector="source/vectors/wikipedia+.d300.w2v.bz2"
perl m@th.[deep].pl -path "/scratch/abarrena014/" -w2v "$vector" -dict source/dict/2014wiki_full_subset.dict.bz2 -cluster 2>> debug/gather.full.m@th.orig.txt

# for LSTM Single Model
# this may take some hours...
vector="source/vectors/wikipedia+.d300.w2v.bz2"
perl m@thNster.[deep].pl -w2v "$vector" -dict source/dict/2014wiki_10occ.dict.bz2 -cluster 2>> debug/gather.m@thNster.txt

# for 500K dataset uncomment and run this:
# vector="source/clusters/wikipedia+.c3000.cls.bz2"
# perl m@th.[deep].pl -path "/scratch/abarrena014/" -augmented -w2v "$vector" -dict source/dict/2014wiki.dict.bz2 -cluster 2>> debug/gather.full.m@th.aug.txt
# vector="source/clusters/wikipedia+.c3000.cls.bz2"
# perl m@th.[deep].pl -path "/scratch/abarrena014/" -w2v "$vector" -dict source/dict/2014wiki.dict.bz2 -cluster 2>> debug/gather.full.m@th.orig.txt


