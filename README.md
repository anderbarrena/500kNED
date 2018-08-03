This repository provides code to replicate the experiments from
"Learning text representations for 500K classification tasks on Named
Entity Disambiguation" by Ander Barrena, Aitor Soroa and Eneko Agirre.

Download data from:
http://ixa2.si.ehu.es/500kNED-download/source.tar.gz and unzip the
data in the main folder.

============================== **Bash Scripts** ============================

 ······························ Main ···································

 **00-do.all.sh**: runs sequentially all scripts bellow to reproduce the
 experiments in "Learning text representations for 500K classification
 tasks on Named Entity Disambiguation" paper.

 ·······································································

 **00-do.gather.sh**: the script creates the training input files for the
 "Word Expert Models" tested on the paper, around 6K mentions (x2 for
 orig and augmented). If you want to create the 500K full dataset,
 change the dictionary file in the script. This may take so much time,
 we recommend to divide the dictionary file and run many processes in
 parallel. Input files stored in ssv/train/mentions/

 **00-do.m@th+.cBoW.sh**: the script trains and tests the Continious Bag
 of Words Word Expert model. Only for the mentions occurring in test
 datasets (Aidatesta, Aidatestb, Tac2010test, Tac2011test and
 Tac2012test datasets). Models are stored in m@ths/ and debug files in
 ssv/train/ and ssv/test/. Output files are stored in m@th/.

 **00-do.m@th+.sBow.sh**: the script trains and tests the Sparse Bag of
 Words Word Expert model. Only for the mentions occurring in test
 datasets (Aidatesta, Aidatestb, Tac2010test, Tac2011test and
 Tac2012test datasets). Models are stored in m@ths/ and debug files in
 ssv/train/ and ssv/test/. Output files are stored in m@th/.

 **00-do.m@th+.lstm.sh**: the script trains and tests the LSTM Word Expert
 model. Only for the mentions occurring in test datasets (Aidatesta,
 Aidatestb, Tac2010test, Tac2011test and Tac2012test datasets). Models
 are stored in m@ths/ and debug files in ssv/train/ and
 ssv/test/. Output files are stored in m@th/.

 **00-do.m@th+.transferLstm.sh**: the script trains and tests the
 transferLSTM Word Expert model. Only for the mentions occurring in
 test datasets (Aidatesta, Aidatestb, Tac2010test, Tac2011test and
 Tac2012test datasets). Models are stored in m@ths/ and debug files in
 ssv/train/ and ssv/test/. Output files are stored in m@th/.

 **01-do.mix.sh**: the script mixes original and augmented model results
 of Word Experts, then evals the system output against the gold
 standard. Eval results in results/ folder.

 **01-do.mix.KB.sh**: the script mixes original and augmented model
 results of Word Experts, and removes those entities not belonging to
 Knowledge Bases (AidaMeans and TacKB). Then evals the system output
 against the gold standard. Eval results are stored in results/
 folder.

========================================================================


============================== **Perl Scripts** ============================

 **m@th.[deep].pl**: Perl script for preproccesing of the Wikipedia data,
 training and testing the "Word Expert Models" (or moths).

 **m@thNster.[deep].pl**: Perl script for preproccesing of the data,
 training and testing the "Single Model" or ([moth]nster).

 **m@th.mxr.pl**: Perl script to mix the results of the "Word Expert
 Models" (original and augmented versions)

========================================================================

============================== **Pytorch Scripts** =========================

 Continious Bag of Words Word Expert pytorch code:
  - @cBoW.aug.py
  - @cBoW.original.py

 LSTM Word Expert pytorch code:
  - @lstm.aug.py
  - @lstm.orig.py

 Sparse Bag of Words Word Expert pytorch code:
  - @sBoW.aug.py
  - @sBoW.original.py

 Transfer LSTM Word Expert pytorch code:
  - @transferLstm.aug.py
  - @transferLstm.orig.py

 Single model LSTM pytorch code:
  - %lstm.py

========================================================================

============================== **Folders** =================================

 **debug**: Contains debugging information from scripts

 **ed**: Contains the preprocessed input and gold standard files from test
 data (Aidatesta, Aidatestb, Tac2010test, Tac2011test and Tac2012test
 datasets).

 **m@th**: Contains the output files of the system for each test dataset

 **m@ths**: Contains the "Word Expert Models" (or moths) and the "Single
 Model" (or [moth]nster) see paper for details

 **results**: Contains the evaluation results

 **source**: Contains preprocessed Wikipedia files
  - clusters: word to cluster files used in sparse experiments
  - dict: name to entity dictionaries
  - entities: contains all the entity context files, one file per
    entity
  - vectors: word vectors used for the continious and LSTM based
    experiments (word2vec)

 **ssv**: Contains the train and test input data (in ssv format) for
 the classifiers and some debugging information

  - train: debuggin files for training and input files in
    train/mentions/ folder
  - test: debuggin and input files for test

========================================================================
