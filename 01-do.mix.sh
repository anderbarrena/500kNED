#!/bin/bash
mkdir -p m@th/cmb
for data in aidatesta aidatestb tac2010test tac2011test tac2012test
do 
    for nn in wikipedia+.d300.w2v.@lstm #wikipedia+.d300.w2v.@cBoW wikipedia+.c3000.cls.@sBoW wikipedia+.d300.w2v.@transferLstm.noGrad wikipedia+.d300.w2v.@transferLstm
    do
	echo $nn
	echo "sigle m@th models"
	# perl m@th.\[deep].pl -out m@th/2014wiki_"$data"_peW."$nn"+.py.1 -dict source/dict/2014wiki_"$data"_subset.dict.bz2 -pes > m@th/cmb/2014wiki_"$data"_peS.prior
	# perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peS.prior
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/2014wiki_"$data"_peW."$nn".orig.1 
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/2014wiki_"$data"_peW."$nn".orig.2 
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/2014wiki_"$data"_peW."$nn".orig.3
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/2014wiki_"$data"_peW."$nn".aug.1 
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/2014wiki_"$data"_peW."$nn".aug.2 
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/2014wiki_"$data"_peW."$nn".aug.3 
	echo "dual m@th-%m@th.aug models"
	perl m@th.mxr.pl  m@th/2014wiki_"$data"_peW."$nn".orig.1 m@th/2014wiki_"$data"_peW."$nn".aug.1 > m@th/cmb/2014wiki_"$data"_peW."$nn".1
	perl m@th.mxr.pl  m@th/2014wiki_"$data"_peW."$nn".orig.2 m@th/2014wiki_"$data"_peW."$nn".aug.2 > m@th/cmb/2014wiki_"$data"_peW."$nn".2
	perl m@th.mxr.pl  m@th/2014wiki_"$data"_peW."$nn".orig.3 m@th/2014wiki_"$data"_peW."$nn".aug.3 > m@th/cmb/2014wiki_"$data"_peW."$nn".3
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".1 
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".2
	perl eval.pl -t 1 gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".3
	rm m@th/cmb/2014wiki_"$data"_peW."$nn".1
	rm m@th/cmb/2014wiki_"$data"_peW."$nn".2
	rm m@th/cmb/2014wiki_"$data"_peW."$nn".3
	echo	
    done
done
