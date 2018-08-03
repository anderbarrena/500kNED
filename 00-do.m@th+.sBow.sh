#!/bin/bash
exit 0

vector=source/clusters/wikipedia+.c3000.cls.bz2
vtag=wikipedia+.c3000.cls
nn='@sBoW.aug.py'
nntag='@sBoW.aug'

for d in plato
do
    perl m@th.[deep].pl -path "/scratch/abarrena014/" -nn "$nn" -w2v "$vector" -augmented -dict source/dict/2014wiki_"$d"_subset.dict.bz2 -train_m0ths 2>> debug/"$d".m@th."$vtag"."$nntag".txt
done

for d in aidatesta aidatestb tac2010test tac2011test tac2012test
do
    for r in 1 2 3
    do
    	perl m@th.[deep].pl -path "/scratch/abarrena014/" -round "$r" -nn "$nn" -augmented -w2v "$vector" -test ed/2014wiki_"$d".words.w50.ed.bz2 -dict source/dict/2014wiki_"$d"_subset.dict.bz2 > m@th/2014wiki_"$d"_peW."$vtag"."$nntag"."$r" 2>> debug/"$d".m@th."$vtag"."$nntag".txt
    done
done

nn='@sBoW.orig.py'
nntag='@sBoW.orig'

for d in plato
do
    perl m@th.[deep].pl -path "/scratch/abarrena014/" -nn "$nn" -w2v "$vector" -dict source/dict/2014wiki_"$d"_subset.dict.bz2 -train_m0ths 2>> debug/"$d".m@th."$vtag"."$nntag".txt
done

for d in aidatesta aidatestb tac2010test tac2011test tac2012test
do
    for r in 1 2 3
    do
    	perl m@th.[deep].pl -path "/scratch/abarrena014/" -round "$r" -nn "$nn" -w2v "$vector" -test ed/2014wiki_"$d".words.w50.ed.bz2 -dict source/dict/2014wiki_"$d"_subset.dict.bz2 > m@th/2014wiki_"$d"_peW."$vtag"."$nntag"."$r" 2>> debug/"$d".m@th."$vtag"."$nntag".txt
    done
done
