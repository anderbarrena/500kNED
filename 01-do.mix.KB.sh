#!/bin/bash
for nn in wikipedia+.d300.w2v.@lstm #wikipedia+.d300.w2v.@cBoW wikipedia+.c3000.cls.@sBoW wikipedia+.d300.w2v.@transferLstm.noGrad wikipedia+.d300.w2v.@transferLstm
do
    for data in aidatesta aidatestb
    do 
    	perl m@th.mxr.pl -dict source/data/2014wiki.dict.aidameans.bz2  m@th/2014wiki_"$data"_peW."$nn".orig.1 m@th/2014wiki_"$data"_peW."$nn".aug.1 > m@th/cmb/2014wiki_"$data"_peW."$nn".1
    	perl m@th.mxr.pl -dict source/data/2014wiki.dict.aidameans.bz2  m@th/2014wiki_"$data"_peW."$nn".orig.2 m@th/2014wiki_"$data"_peW."$nn".aug.2 > m@th/cmb/2014wiki_"$data"_peW."$nn".2
    	perl m@th.mxr.pl -dict source/data/2014wiki.dict.aidameans.bz2  m@th/2014wiki_"$data"_peW."$nn".orig.3 m@th/2014wiki_"$data"_peW."$nn".aug.3 > m@th/cmb/2014wiki_"$data"_peW."$nn".3
    	perl ~/EDL/ebaluatzeko_tresnak/upperbound.pl -t 1 ~/EDL/erauzitako_datasetak/gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".1
    	perl ~/EDL/ebaluatzeko_tresnak/upperbound.pl -t 1 ~/EDL/erauzitako_datasetak/gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".2 
    	perl ~/EDL/ebaluatzeko_tresnak/upperbound.pl -t 1 ~/EDL/erauzitako_datasetak/gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".3 
    	rm m@th/cmb/2014wiki_"$data"_peW."$nn".1
    	rm m@th/cmb/2014wiki_"$data"_peW."$nn".2
    	rm m@th/cmb/2014wiki_"$data"_peW."$nn".3
    	echo	
    done

    for data in tac2010test tac2011test tac2012test
    do 
	perl m@th.mxr.pl -kb source/data/2014wiki.tacKB.ents.bz2  m@th/2014wiki_"$data"_peW."$nn".orig.1 m@th/2014wiki_"$data"_peW."$nn".aug.1 > m@th/cmb/2014wiki_"$data"_peW."$nn".1
	perl m@th.mxr.pl -kb source/data/2014wiki.tacKB.ents.bz2  m@th/2014wiki_"$data"_peW."$nn".orig.2 m@th/2014wiki_"$data"_peW."$nn".aug.2 > m@th/cmb/2014wiki_"$data"_peW."$nn".2
	perl m@th.mxr.pl -kb source/data/2014wiki.tacKB.ents.bz2  m@th/2014wiki_"$data"_peW."$nn".orig.3 m@th/2014wiki_"$data"_peW."$nn".aug.3 > m@th/cmb/2014wiki_"$data"_peW."$nn".3
	perl ~/EDL/ebaluatzeko_tresnak/upperbound.pl -t 1 ~/EDL/erauzitako_datasetak/gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".1
	perl ~/EDL/ebaluatzeko_tresnak/upperbound.pl -t 1 ~/EDL/erauzitako_datasetak/gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".2
	perl ~/EDL/ebaluatzeko_tresnak/upperbound.pl -t 1 ~/EDL/erauzitako_datasetak/gs/"$data"_wiki2014.key m@th/cmb/2014wiki_"$data"_peW."$nn".3
	rm m@th/cmb/2014wiki_"$data"_peW."$nn".1
	rm m@th/cmb/2014wiki_"$data"_peW."$nn".2
	rm m@th/cmb/2014wiki_"$data"_peW."$nn".3
	echo	
    done
done
