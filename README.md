
This repository provides code to replicate the experiments from
"Learning text representations for 500K classification tasks on Named
Entity Disambiguation" by Ander Barrena, Aitor Soroa and Eneko Agirre.
Word Expert models are named as moths (or m@ths) in the code. 

Download data from:

  http://ixa2.si.ehu.es/500kNED-download/source.tar.gz Unzip the
  data in the main folder. Contains dictionary, vector and cluster
  files.

  http://ixa2.si.ehu.es/500kNED-download/entities.tar.gz Unzip the
  data in ssv/train/ folder. Contains all the prepocessed contexts for
  all the entities in Wikipedia (06Nov2014 dump)

# **Bash Scripts**

 ************************************************************************
 **00-do.all.sh**: runs sequentially all scripts bellow to reproduce the
 experiments in "Learning text representations for 500K classification
 tasks on Named Entity Disambiguation" paper.
 ************************************************************************

 **00-do.gather.sh**: the script creates the training input files for
 the "Word Expert Models" tested on the paper (Aidatesta, Aidatestb,
 Tac2010test, Tac2011test and Tac2012test datasets), around 6K
 mentions. If you want to create the 500K full dataset, uncomment the
 last lines in the script. This may take so much time, we recommend to
 divide the dictionary file and run many processes in
 parallel. Training input files stored in ssv/train/mentions/

   The trainining data is stored as:
   * one file per mention in the dictionary
   * in the first line, the entity priors (not used) 
   * the rest of the lines gather all the mention contexts 
   * the contexts are represented as the 100 non stop words
     sorrounding the mention
   * the words are replaced by their corresponding index in the
     vectors file /source/vectors/wikipedia.d300.w2v.bz2

 **00-do.m@th.cBoW.sh**: the script trains and tests the Continious Bag
 of Words Word Expert model. Only for the mentions occurring in test
 datasets. Models are stored in m@ths/ and debug files in
 ssv/train/ and ssv/test/. Output files are stored in m@th/.

 **00-do.m@th.sBow.sh**: the script trains and tests the Sparse Bag of
 Words Word Expert model. Only for the mentions occurring in test
 datasets. Models are stored in m@ths/ and debug files in
 ssv/train/ and ssv/test/. Output files are stored in m@th/.

 **00-do.m@th.lstm.sh**: the script trains and tests the LSTM Word Expert
 model. Only for the mentions occurring in test datasets. Models
 are stored in m@ths/ and debug files in ssv/train/ and
 ssv/test/. Output files are stored in m@th/.

 **00-do.m@th.lstm.SingleModel.sh**: the script trains the LSTM
 Single model. Models are stored in m@ths/ and debug files in
 ssv/train/. Output files are stored in m@th/.

 **00-do.m@th.transferLstm.sh**: the script trains and tests the
 transferLSTM Word Expert model. Only for the mentions occurring in
 test datasets. Models are stored in m@ths/ and debug files in
 ssv/train/ and ssv/test/. Output files are stored in m@th/.
 
 **01-do.mix.sh**: the script mixes original and augmented model results
 of Word Experts, then evals the system output against the gold
 standard. Eval results are stored in results/ folder.

 **01-do.mix.KB.sh**: the script mixes original and augmented model
 results of Word Experts, and removes those entities not belonging to
 Knowledge Bases (AidaMeans and TacKB). Then evals the system output
 against the gold standard. Eval results are stored in results/
 folder.

# **Perl Scripts** 

 **m@th.[deep].pl**: Perl script for preproccesing Wikipedia, it
 creates the training dataset, abd used for training and testing the
 "Word Expert Models" (or moths).

 **m@thNster.[deep].pl**: Perl script for preproccesing of the data,
 training and testing the "Single Model" or ([moth]nster).

 **m@th.mxr.pl**: Perl script to mix the results of the "Word Expert
 Models" (original and augmented versions)

# **Pytorch Scripts**

 Continious Bag of Words Word Expert pytorch code:			       
  - @cBoW.py 

 LSTM Word Expert pytorch code:			       
  - @lstm.py

 Sparse Bag of Words Word Expert pytorch code:			       
  - @sBoW.py

 Transfer LSTM Word Expert pytorch code:
  - @transferLstm.py

 Single model LSTM pytorch code:
  - %lstm.py

# **Folders**
 
 Some of the folders will be created or filled during execution:

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
  - entities: contains entity context files, one file per entity
  - vectors: word vectors used for the continious and LSTM based
    experiments (word2vec)

 **ssv**: Contains the train and test input data (in ssv format) for
 the classifiers and some debugging information

  - train: debuggin files for training and input files in
    train/mentions/ folder
  - test: debuggin and input files for test
